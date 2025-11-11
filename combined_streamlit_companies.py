import streamlit as st
import pandas as pd
import json
import os
import time
import re
import requests
from bs4 import BeautifulSoup
from collections import Counter

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
# from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate


# --- CONFIGURATION ---
LOCAL_DATA_FILE = "data/merged_data.csv"
CHROMA_DB_PATH = "chroma_db"

# --- NEW: MODEL CONFIGURATION ---
MODEL_CONFIG = {
    "gpt-4o": {
        "display_name": "GPT-4o (Fast & Balanced)",
        "cost_input_per_m": 5.00,
        "cost_output_per_m": 15.00,
    },
    "gpt-4-turbo": {
        "display_name": "GPT-4 Turbo (Most Powerful)",
        "cost_input_per_m": 10.00,
        "cost_output_per_m": 30.00,
    },
    "gpt-3.5-turbo": {
        "display_name": "GPT-3.5 Turbo (Most Economical)",
        "cost_input_per_m": 0.50,
        "cost_output_per_m": 1.50,
    },
}
# ==============================================================================
# SECTION 1: HELPER FUNCTIONS (Consolidated from both scripts)
# ==============================================================================

@st.cache_data
def load_data(file_path):
    """
    MODIFIED: Loads data and creates a company-to-product mapping.
    """
    try:
        merged_data = pd.read_csv(file_path)
        merged_data['doc_text'] = merged_data['doc_text'].fillna('')
        merged_data['doc_text'] = merged_data['doc_text'].astype(str).apply(
            lambda x: x.encode('ascii', 'ignore').decode('ascii')
        )

        required_cols = ['Product', 'Company', 'doc_text']
        if not all(col in merged_data.columns for col in required_cols):
             st.error(f"Data is missing one of the required columns: {required_cols}. Found: {merged_data.columns.tolist()}")
             return None, None, None, None

        merged_data = merged_data[required_cols]

        # 1. Get Product Options (original - still needed for P2 full list)
        unique_product_names = sorted(merged_data['Product'].unique().tolist())
        product_options = ["-- Select a Product --"] + unique_product_names

        # 2. NEW: Get Company Options
        unique_company_names = sorted(merged_data['Company'].unique().tolist())
        company_options = ["-- Select a Company --"] + unique_company_names

        # 3. NEW: Create Company-to-Product Map
        company_to_products_map = {}
        # Pre-populate the map for all companies
        for company in unique_company_names:
            products = sorted(merged_data[merged_data['Company'] == company]['Product'].unique().tolist())
            # Add a "select" option for each list
            company_to_products_map[company] = ["-- Select a Product --"] + products

        # Add a default entry for the main "Select a Company" option
        company_to_products_map["-- Select a Company --"] = ["-- Select a Company First --"]

        # Return all new items
        return merged_data, product_options, company_options, company_to_products_map

    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        return None, None, None, None

# --- Pipeline 1 Load Vector Store ---
@st.cache_resource
def load_vectorstore(api_key):
    """Loads the Chroma vector store from disk."""
    st.info("Loading Chroma vector store from disk... (This runs only once per session)")
    try:
        if not os.path.exists(CHROMA_DB_PATH):
            st.error(f"Vector store directory not found at '{CHROMA_DB_PATH}'. Please ensure the notebook's initial cells have been run to create it.")
            return None

        if not api_key:
             st.error("Cannot load vector store: OpenAI API Key is missing.")
             return None

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)

        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings
        )
        st.success(f"Vector store loaded successfully from '{CHROMA_DB_PATH}'.")
        return vectorstore
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None

def calculate_llm_cost_p1(input_tokens_per_call, output_tokens_per_call, num_calls, model_name):
    """Estimates the max token usage and cost for API calls in Pipeline 1."""
    # MODIFIED: Use dynamic costs from MODEL_CONFIG
    model_pricing = MODEL_CONFIG.get(model_name)
    if not model_pricing:
        st.error(f"Pricing for model '{model_name}' not found. Defaulting to gpt-4o pricing.")
        model_pricing = MODEL_CONFIG["gpt-4o"]

    COST_INPUT_PER_M = model_pricing['cost_input_per_m']
    COST_OUTPUT_PER_M = model_pricing['cost_output_per_m']

    total_input_tokens = input_tokens_per_call * num_calls
    total_output_tokens = output_tokens_per_call * num_calls

    input_cost = (total_input_tokens / 1_000_000) * COST_INPUT_PER_M
    output_cost = (total_output_tokens / 1_000_000) * COST_OUTPUT_PER_M

    total_cost = input_cost + output_cost
    return total_input_tokens, total_output_tokens, total_cost

def estimate_pipeline_cost_p1(api_key, merged_data, target_product_input, num_angles, model_name):
    """
    Calculates the estimated cost for the entire Pipeline 1 run.
    (This is the ORIGINAL product-to-product logic)
    """
    # This correctly uses the new 'merged_data' dataframe
    target_product_row = merged_data.loc[merged_data['Product'] == target_product_input].iloc[0]
    target_doc_length = len(target_product_row['doc_text'].split())
    target_doc_tokens = int(target_doc_length * 1.33) # Convert word count to estimated tokens

    # 1. Estimate Stage 1 (LLM 1 - Strategist)
    llm1_input_tokens = 500 + target_doc_tokens
    llm1_input, llm1_output, llm1_cost = calculate_llm_cost_p1(llm1_input_tokens, 300, 1, model_name)

    # 2. Estimate Stage 2 (LLM 2 - Profiler)
    llm2_input_tokens = 800 + target_doc_tokens
    llm2_input, llm2_output, llm2_cost = calculate_llm_cost_p1(llm2_input_tokens, 300, num_angles, model_name)

    # 3. Estimate Stage 4 (LLM 3 - Scorer)
    N_c_estimate = max(5, int(num_angles * 2.5)) # Conservative estimate for number of candidates
    llm3_input_tokens = 3500 + target_doc_tokens # (Base estimate for prompt + chunks) + target doc
    llm3_input, llm3_output, llm3_cost = calculate_llm_cost_p1(llm3_input_tokens, 200, N_c_estimate, model_name)

    # Total Estimate
    total_tokens = llm1_input + llm1_output + llm2_input + llm2_output + llm3_input + llm3_output
    total_cost = llm1_cost + llm2_cost + llm3_cost

    return total_tokens, total_cost, N_c_estimate


def calculate_llm_cost_p2(prompt_template_string, product_doc_1, product_doc_2, model_name):
    """Estimates the max token usage and cost for the API call in Pipeline 2."""
    # MODIFIED: Use dynamic costs from MODEL_CONFIG
    model_pricing = MODEL_CONFIG.get(model_name)
    if not model_pricing:
        st.error(f"Pricing for model '{model_name}' not found. Defaulting to gpt-4o pricing.")
        model_pricing = MODEL_CONFIG["gpt-4o"]

    COST_INPUT_PER_M = model_pricing['cost_input_per_m']
    COST_OUTPUT_PER_M = model_pricing['cost_output_per_m']
    output_tokens = 1800

    try:
        temp_prompt = prompt_template_string.replace(
            '{product_1_doc}', product_doc_1
        ).replace(
            '{product_2_doc}', product_doc_2
        )
        input_text = temp_prompt.replace(
            '{data_source_label}', 'Internal Data + External Web Data'
        )

    except Exception as e:
        st.warning(f"Cost calculation substitution failed: {e}. Using estimated default word count.")
        input_text = product_doc_1 + product_doc_2

    input_words = len(input_text.split())
    input_tokens = int(input_words * 1.33)

    input_cost = (input_tokens / 1_000_000) * COST_INPUT_PER_M
    output_cost = (output_tokens / 1_000_000) * COST_OUTPUT_PER_M

    total_cost = input_cost + output_cost

    return input_tokens, output_tokens, total_cost

def extract_website_url(doc_text: str) -> str:
    """Extracts the website URL using pure Python regex."""
    match = re.search(r"URLs_y:\s*website\s*-\s*([^;,\s]+)", doc_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

def get_external_data_snippets(website_url: str) -> str:
    """Fetches the maximum amount of raw, visible text content from the given URL."""
    if not website_url:
        return f"*** EXTERNAL WEB DATA (FAILURE) ***\n(No website URL available for scraping.)"

    if not re.match(r'http(s)?://', website_url, re.IGNORECASE):
        checked_url = 'https://' + website_url
    else:
        checked_url = website_url

    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(checked_url, timeout=20, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        for tag in soup(['script', 'style']):
            tag.decompose()

        full_raw_text = soup.body.get_text(separator='\n', strip=True) if soup.body else soup.get_text(separator='\n', strip=True)

        cleaned_text = re.sub(r'[\r\n]+', '\n', full_raw_text)
        cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)

        if len(cleaned_text.split()) < 10:
             return f"*** EXTERNAL WEB DATA (FAILURE) ***\nOnly minimal text found ({len(cleaned_text.split())} words). Site may be entirely dynamic."

        return (
            f"*** EXTERNAL WEB DATA (MAXIMUM RAW CONTENT FROM {checked_url}) ***\n"
            f"--- START OF RAW DATA ---\n"
            f"{cleaned_text}"
        )

    except requests.exceptions.RequestException as e:
        return f"*** EXTERNAL WEB DATA (FAILURE) ***\nRequest failed for {checked_url}. Reason: {e}"
    except Exception as e:
        return f"*** EXTERNAL WEB DATA (FAILURE) ***\nAn unexpected error occurred during parsing: {e}"

# ==============================================================================
# SECTION 2: LLM CHAIN DEFINITIONS (P1 & P2)
# ==============================================================================

# --- Pipeline 1 LLM Chains ---
def define_llm_chains_p1(api_key, num_angles, model_name):
    """Defines all three LLM chains for Pipeline 1."""
    # MODIFIED: Use dynamic model_name
    llm = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key=api_key)

    # --- LLM 1: The Dynamic Strategist (PROMPT UPDATED from pipeline1_streamlit.py) ---
    strategist_prompt_template = f"""
    You are a top-tier product strategist. Your goal is to identify broad categories of highly synergistic **PRODUCT TYPES** (service, technology, or data products) for the 'Target Product' below.
    Analyze the 'Target Product' and its features, then devise **{num_angles}** most potent partnership pillars. Each pillar MUST describe a **PRODUCT TYPE**.
    For each pillar, write a paragraph focusing on the ideal synergistic **PRODUCT's features, function, and data**, and how they would enhance the Target Product.
    --- TARGET PRODUCT ---
    {{target_doc}}
    --- OUTPUT FORMAT ---
    Provide your response as a single, valid JSON object with exactly **{num_angles}** key-value pairs.
    """
    strategist_prompt = ChatPromptTemplate.from_template(strategist_prompt_template)
    strategist_chain = strategist_prompt | llm | JsonOutputParser()

    # --- LLM 2: The Profiler (PROMPT UPDATED from pipeline1_streamlit.py) ---
    profiler_prompt_template = """
    You are a brilliant business strategist. Based on the following 'Partnership Strategy', generate a rich, abstract profile of an ideal partner product.
    Do not invent a name for a product. Describe its features, target users, and core value proposition in detail.
    --- PARTNERSHIP STRATEGY ---
    {strategy_description}
    --- TARGET PRODUCT ---
    {target_doc}
    """
    profiler_prompt = ChatPromptTemplate.from_template(profiler_prompt_template)
    profiler_chain = profiler_prompt | llm | StrOutputParser()

    # --- LLM 3: The Initial Scorer (PROMPT UPDATED from pipeline1_streamlit.py) ---
    scorer_prompt_template = """
    You are a rapid-assessment business analyst. Assess the synergy potential between the 'Target Product' and the 'Potential Partner' based ONLY on the provided texts.
    Provide a single score from 1 (low synergy) to 10 (high synergy) and a one-sentence justification.
    --- TARGET PRODUCT ---
    {target_doc}
    --- POTENTIAL PARTNER CHUNKS ---
    {candidate_chunks_text}
    --- OUTPUT FORMAT ---
    Provide your response as a JSON object with two keys: "score" and "reasoning".
    """
    scorer_prompt = ChatPromptTemplate.from_template(scorer_prompt_template)
    scorer_chain = scorer_prompt | llm | JsonOutputParser()

    return strategist_chain, profiler_chain, scorer_chain

# --- Pipeline 2 LLM Chain ---
def define_analyst_chain_p2(api_key, prompt_template, model_name):
    """Defines the heavy LLM chain for detailed synergy analysis (LLM 3)."""
    # MODIFIED: Use dynamic model_name
    llm = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key=api_key)
    analyst_prompt = ChatPromptTemplate.from_template(prompt_template)
    return analyst_prompt | llm | StrOutputParser()

# ==============================================================================
# SECTION 3: PIPELINE 1 EXECUTION LOGIC (Adapted for live output)
# ==============================================================================

@st.cache_data(show_spinner=False)
def run_pipeline_execution_p1(api_key, merged_data, target_product_input, num_angles, _vectorstore, model_name):
    """
    Executes the core RAG pipeline (P1) and returns the final ranked shortlist dataframe
    and the intermediate candidate data frame.
    (This is the ORIGINAL product-to-product logic)
    """
    # 1. SETUP: Create the normalized target name for RAG self-filtering
    # This must match the format of 'Product' in the NEW Chroma DB metadata
    target_name_normalized = re.sub(r'[^a-zA-Z0-9]', '', target_product_input.lower())

    # Use 'Product' column (from new merged_data) for locating the product row
    target_product_row = merged_data.loc[merged_data['Product'] == target_product_input].iloc[0]
    target_product_description = target_product_row['doc_text']

    # MODIFIED: Pass model_name to chain definition
    strategist_chain, profiler_chain, scorer_chain = define_llm_chains_p1(api_key, num_angles, model_name)

    # --- STAGE 1: Dynamic Strategy Generation (LLM 1) ---
    st.subheader(f"1️⃣ Dynamic Strategy Generation (LLM 1) - {num_angles} Pillars")
    with st.spinner("🧠 Analyzing product and defining strategic pillars..."):
        # Use the cleaned doc_text (ASCII only) for the LLM
        clean_target_doc = target_product_description.encode("ascii", "ignore").decode("ascii")
        synergy_strategies = strategist_chain.invoke({"target_doc": clean_target_doc})
        st.success("Strategic Pillars Generated.")
        st.json(synergy_strategies)

    st.markdown("---")

    # --- STAGE 2 & 3: Profile, Retrieve, and Aggregate Candidates (LLM 2 + RAG) ---
    st.subheader("2️⃣ Candidate Profiling and RAG Retrieval (LLM 2 + Chroma)")
    all_retrieved_chunks = []
    # Key now uses (Company, Product) to match the NEW RAG metadata schema
    candidate_product_summary = {}

    def custom_retriever(query_text: str):
        return _vectorstore.similarity_search(query_text, k=8)

    progress_bar = st.progress(0, text="Starting profiling and retrieval...")

    for i, (strategy_type, strategy_desc) in enumerate(synergy_strategies.items()):
        progress_bar.progress((i + 1) / (num_angles + 1), text=f"Profiling & Retrieving for: '{strategy_type}'...")

        # LLM 2: Profiler
        hypothetical_doc = profiler_chain.invoke({
            "strategy_description": strategy_desc,
            "target_doc": clean_target_doc # Use clean doc
        })

        # RAG: Retriever
        retrieved_docs = custom_retriever(hypothetical_doc)

        # --- FIX: Self-filtering now uses the NEW 'Product' metadata key ---
        filtered_docs = []
        for doc in retrieved_docs:
            # Use 'Product' metadata key, normalize it for self-filtering
            product_meta = doc.metadata.get('Product', '')
            if product_meta != target_product_input: # Simpler check
                filtered_docs.append(doc)
        # --- END FIX ---

        all_retrieved_chunks.extend(filtered_docs)

        for doc in filtered_docs:
            # --- FIX: Consistently use the NEW metadata key 'Product' for product name retrieval ---
            product_name_meta = doc.metadata.get('Product')
            company_name_meta = doc.metadata.get('Company') # This was already correct

            # Ensure data is valid before using as key
            if product_name_meta is None or company_name_meta is None:
                 continue

            product_key = (company_name_meta, product_name_meta)

            candidate_product_summary.setdefault(product_key, {
                'total_chunks': 0,
                'angles': Counter()
            })
            candidate_product_summary[product_key]['total_chunks'] += 1
            candidate_product_summary[product_key]['angles'][strategy_type] += 1

    progress_bar.progress(1.0, text="Completed profiling and retrieval.")
    time.sleep(1)
    progress_bar.empty()

    sorted_candidates = sorted(candidate_product_summary.items(), key=lambda item: item[1]['total_chunks'], reverse=True)
    unique_candidates_keys = [key for key, value in sorted_candidates]
    st.success(f"Found a total of **{len(sorted_candidates)}** unique potential partners from **{len(all_retrieved_chunks)}** retrieved chunks.")

    # Prepare Intermediate Candidates Data Frame
    intermediate_candidate_data = []
    for company, product_name in unique_candidates_keys:
        stats = candidate_product_summary[(company, product_name)]
        angle_breakdown = ", ".join([f"{angle.replace('_', ' ').title()} ({count})" for angle, count in stats['angles'].items()])
        intermediate_candidate_data.append({
            "Product": product_name,  # Value from 'Product' metadata
            "Company": company,
            "Total Chunks": stats['total_chunks'],
            "Source Angles": angle_breakdown
        })
    df_intermediate = pd.DataFrame(intermediate_candidate_data).reset_index(names=['Rank (by chunk count)'])

    # --- Display Intermediate Candidate List (LLM 2 + RAG Output) ---
    st.markdown("### Intermediate Candidate List (Ranked by Chunk Count)")
    if intermediate_candidate_data:
        st.dataframe(df_intermediate, use_container_width=True, hide_index=True)
    else:
         st.warning("No candidates were found in the RAG retrieval step.")
    # --- END Display Fix ---

    # --- STAGE 4: Batch Initial Synergy Analysis and Ranking (LLM 3) ---
    st.markdown("---")
    st.subheader("3️⃣ Automated Initial Synergy Ranking (LLM 3)")

    all_initial_results = []
    product_chunks_map = {}
    for doc in all_retrieved_chunks:
        # --- FIX: RAG METADATA ACCESS: MUST use 'Product' and 'Company' ---
        product_name_meta = doc.metadata.get('Product')
        company_name_meta = doc.metadata.get('Company')

        if product_name_meta is None or company_name_meta is None:
            continue

        product_key = (company_name_meta, product_name_meta)
        product_chunks_map.setdefault(product_key, []).append(doc.page_content)

    num_candidates = len(unique_candidates_keys)
    ranking_progress = st.progress(0, text=f"Starting initial analysis of {num_candidates} candidates...")

    for i, (company, product_name) in enumerate(unique_candidates_keys):
        ranking_progress.progress((i + 1) / num_candidates, text=f"Analyzing candidate {i+1}/{num_candidates}: {product_name}...")

        candidate_chunks = product_chunks_map.get((company, product_name), [])
        candidate_chunks_text = "\n\n---\n\n".join(candidate_chunks)

        # Safegard for non-ASCII characters (kept from original)
        candidate_chunks_text_clean = candidate_chunks_text.encode("ascii", "ignore").decode("ascii")

        initial_analysis = scorer_chain.invoke({
            "target_doc": clean_target_doc, # Use clean doc
            "candidate_chunks_text": candidate_chunks_text_clean
        })

        score = initial_analysis.get('score', 0)
        reasoning = initial_analysis.get('reasoning', 'No reasoning provided.')

        all_initial_results.append({
            "Rank": 0,
            "Score": score,
            "Product": product_name, # Value from 'Product' metadata
            "Company": company,
            "Justification": reasoning
        })
        time.sleep(0.1)

    ranking_progress.progress(1.0, text="Analysis complete.")
    time.sleep(1)
    ranking_progress.empty()
    st.success("All candidates analyzed.")

    # --- Final Ranked Shortlist Output ---
    ranked_shortlist = sorted(all_initial_results, key=lambda x: x['Score'], reverse=True)

    for i, result in enumerate(ranked_shortlist):
        result['Rank'] = i + 1

    df_final = pd.DataFrame(ranked_shortlist)

    # Return both dataframes and the strategist output to be stored
    return df_final, df_intermediate, synergy_strategies

# ==============================================================================
# SECTION 4: PIPELINE 2 EXECUTION LOGIC (Adapted from original script)
# ==============================================================================

# --- DEFAULT EDITABLE INSTRUCTIONS (Does NOT include placeholders) ---
DEFAULT_EDITABLE_INSTRUCTIONS = """
You are a senior partnership analyst for digital health companies, with decades of experience identifying intricate, high-value partnership opportunities. Your task is to conduct a deep-dive analysis of the synergy potential between 'Product 1' and 'Product 2'. Go beyond surface-level similarities and think like a creative business strategist to uncover the non-obvious ways these companies could create exponential value together.

First, read and internalize the data for both companies. Then, structure your analysis as an **Executive Partnership Briefing** with the following sections:

**1. Executive Summary:**
Start with a concise, top-level paragraph summarizing the core synergy thesis. What is the single most compelling reason this partnership could work?

**2. Key Synergy Opportunities (The 'Why'):**
In a detailed, bullet-pointed list, identify the most significant opportunities for synergy. For each point, explain the underlying \\"why\\". Consider questions like:
- **Value Chain Integration:** Can one company's product fill a critical gap in the other's customer value chain?
- **Data & Insights Synergy:** Could their combined data assets unlock new insights, products, or efficiencies?
- **Complementary Go-to-Market:** Beyond simple market access, do their sales motions or brand positions complement each other in a unique way?
- **Unlocking New Business Models:** Could a partnership enable a completely new product, service, or pricing model that neither could achieve alone?

**3. Potential Risks & Mitigations:**
In a detailed, bullet-pointed list, identify potential risks or challenges in this partnership (e.g.: conflicting company cultures, channel conflict, technological integration challenges). For each risk, suggest a potential mitigation strategy.

**4. Seven-Axis Synergy Scores:**
Provide a score from 1 (low synergy) to 10 (high synergy) for each of the following seven axes. For each score, provide a brief justification based on the provided data.
- **Clinical & Operational Alignment:** <Score and Justification> (Considers clinical pathways, TAT, and disease specificity match)
- **Patient Experience & Journey Continuity:** <Score and Justification> (Considers care continuity and patient drop-off risk)
- **Financial & Reimbursement Compatibility:** <Score and Justification> (Considers insurance coverage and out-of-pocket cost alignment)
- **Technological Interoperability:** <Score and Justification> (Considers APIs, EHR/EMR systems, and data standards)
- **Regulatory & Compliance Alignment:** <Score and Justification> (Considers overlap in regulatory pathways and certifications)
- **Geographic & Logistics Fit:** <Score and Justification> (Considers specimen logistics and physical location compatibility)
- **Strategic & Innovation Potential:** <Score and Justification> (Considers co-development opportunity and market expansion complementarity)

**5. Final Recommendation:**
Conclude with a single, direct recommendation: \\"Highly Recommended\\\", \\"Recommended with Reservations\\\", or \\"Not Recommended\\".
"""

# --- FIXED PLACEHOLDERS (Always appended to the end of the instructions) ---
FIXED_PLACEHOLDERS = """
--- PRODUCT 1 ({data_source_label}) ---
{product_1_doc}

--- PRODUCT 2 ({data_source_label}) ---
{product_2_doc}
"""

def get_data_for_display(doc_content, is_web_mode):
    """Helper to parse the combined doc for display in P2 review stage."""
    if is_web_mode:
        if "*** Internal Data ***" in doc_content:
            try:
                internal = doc_content.split("*** Internal Data ***\n")[1].split("\n\n*** External Web Data ***")[0].strip()
                external = doc_content.split("\n\n*** External Web Data ***\n")[1].strip()
            except IndexError:
                internal = "Parsing Error: Check Raw Document Structure."
                external = doc_content
        else:
             internal = doc_content
             external = "N/A (Markers missing)"
        return internal, external
    else:
        # For Internal Data Only mode, the entire doc content is the internal data
        internal = doc_content.rpartition("---")[2].strip()
        return internal, None


# ==============================================================================
# SECTION 5: STREAMLIT APP STRUCTURE AND ROUTING
# ==============================================================================

st.set_page_config(layout="wide", page_title="HealthLab Synergy Detection")
st.title("HealthLab Synergy Detection for Digital Health Companies")
st.sidebar.title("App Navigation")

# --- Initialize Session State for Routing and Data Storage ---
if 'page' not in st.session_state: st.session_state.page = 'p1'
if 'api_key' not in st.session_state: st.session_state.api_key = ""
# NEW: Initialize selected model
if 'selected_model' not in st.session_state: st.session_state.selected_model = "gpt-4o"

# Pipeline 1 State (Resetting stage for cost confirmation flow)
if 'pipeline_stage_p1' not in st.session_state: st.session_state.pipeline_stage_p1 = 'setup' # 'setup', 'estimated', 'running', 'complete'
if 'final_results_p1' not in st.session_state: st.session_state.final_results_p1 = None
if 'estimated_cost_data_p1' not in st.session_state: st.session_state.estimated_cost_data_p1 = None


# Pipeline 2 State
if 'collected_data_p2' not in st.session_state: st.session_state.collected_data_p2 = None
if 'analysis_ready_p2' not in st.session_state: st.session_state.analysis_ready_p2 = False
if 'web_data_toggle_p2' not in st.session_state: st.session_state.web_data_toggle_p2 = False
if 'editable_prompt_p2' not in st.session_state: st.session_state.editable_prompt_p2 = DEFAULT_EDITABLE_INSTRUCTIONS
if 'analysis_report_p2' not in st.session_state: st.session_state.analysis_report_p2 = None


# Load shared data
if not os.path.exists(LOCAL_DATA_FILE):
    st.error(f"Data file not found at '{LOCAL_DATA_FILE}'. Please ensure the file is in the correct location.")
    st.stop()

# MODIFIED: Unpack new outputs from load_data
merged_data, product_options, company_options, company_to_products_map = load_data(LOCAL_DATA_FILE)
if merged_data is None:
    st.error("Data loading failed. Please check the file and column names.")
    st.stop()

# --- Sidebar Navigation Buttons ---
if st.sidebar.button("💡 Synergy Filtering (Pipeline 1)", use_container_width=True):
    st.session_state.page = 'p1'
    st.rerun()
if st.sidebar.button("🔎 1-on-1 Deep Dive (Pipeline 2)", use_container_width=True):
    st.session_state.page = 'p2'
    st.rerun()

st.sidebar.markdown("---")
# --- Global API Key Input (Source of Truth) ---
st.session_state.api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    value=st.session_state.api_key,
    key='global_api_key_input',
    help="Used for all LLM calls across both pipelines."
)

# --- NEW: Global Model Selection ---
model_display_names = [v['display_name'] for v in MODEL_CONFIG.values()]
model_internal_names = list(MODEL_CONFIG.keys())
# Find the index of the current model to set the selectbox default
try:
    current_model_index = model_internal_names.index(st.session_state.selected_model)
except ValueError:
    current_model_index = 0 # Default to the first model if not found

st.session_state.selected_model = st.sidebar.selectbox(
    "Select LLM Model",
    options=model_internal_names,
    index=current_model_index,
    format_func=lambda x: MODEL_CONFIG[x]['display_name'],
    key='global_model_select',
    help="Determines the model used for all analysis and cost calculations."
)

st.sidebar.markdown("---")

# --- Centralized Reset Buttons ---

# Function to reset P1 state
def reset_pipeline_1_state():
    st.session_state.pipeline_stage_p1 = 'setup'
    st.session_state.final_results_p1 = None
    st.session_state.estimated_cost_data_p1 = None
    # NEW: Reset P1 dropdowns
    if 'p1_target_company_select' in st.session_state: del st.session_state['p1_target_company_select']
    if 'p1_target_product_select' in st.session_state: del st.session_state['p1_target_product_select']
    # Do NOT clear p1_num_angles slider value or model selection

# Function to reset P2 state
def reset_pipeline_2_state():
    st.session_state.collected_data_p2 = None
    st.session_state.analysis_ready_p2 = False
    st.session_state.analysis_report_p2 = None
    st.session_state.editable_prompt_p2 = DEFAULT_EDITABLE_INSTRUCTIONS
    # NEW: Reset P2 dropdowns
    if 'p2_company_1_select' in st.session_state: del st.session_state['p2_company_1_select']
    if 'product_1_select_p2' in st.session_state: del st.session_state['product_1_select_p2']
    if 'p2_company_2_select' in st.session_state: del st.session_state['p2_company_2_select']
    if 'product_2_select_p2' in st.session_state: del st.session_state['product_2_select_p2']


# Button 1: Reset Global API Key (Resets API for both pipelines)
if st.sidebar.button("Reset API Key", type="secondary", key='global_api_reset'):
    st.session_state.api_key = ""
    if 'global_api_key_input' in st.session_state:
        st.session_state['global_api_key_input'] = ""
    st.rerun()

# Button 2: Reset Active Pipeline
pipeline_reset_label = "Reset Pipeline 1 (Synergy Filtering)" if st.session_state.page == 'p1' else "Reset Pipeline 2 (1-on-1 Deep Dive)"

if st.sidebar.button(pipeline_reset_label, type="secondary", key='active_pipeline_reset'):
    if st.session_state.page == 'p1':
        reset_pipeline_1_state()
    elif st.session_state.page == 'p2':
        reset_pipeline_2_state()
    st.rerun()

# ==============================================================================
# VIEW 1: PIPELINE 1 - SYNERGY FILTERING
# ==============================================================================
if st.session_state.page == 'p1':
    st.header("💡 Pipeline 1: RAG-Based Partnership Synergy Filtering")
    st.markdown("Use this stage to quickly identify the best candidates for a selected **Target Product**.")

    # --- Sidebar Configuration for P1 ---
    with st.sidebar:
        st.header("Pipeline 1 Config")
        num_angles = st.slider(
            "Number of Synergy Angles (N)",
            min_value=1,
            max_value=5,
            value=2,
            step=1,
            key='p1_num_angles',
            help="LLM 1 will generate N distinct strategic pillars to guide the search."
        )

        # --- MODIFIED: Dependent Dropdowns ---
        target_company_select = st.selectbox(
            "Select Target Company",
            company_options,
            index=0,
            key='p1_target_company_select'
        )

        # Get the product list based on the selected company
        available_products = company_to_products_map.get(target_company_select, [])

        target_product_input = st.selectbox(
            "Select Target Product",
            available_products, # <-- This list is now dynamic
            index=0,
            key='p1_target_product_select' # <-- Use original key
        )
        # --- END MODIFICATION ---

        # Main Action Button Logic (Now just estimates cost and sets stage)
        if st.button("Estimate Cost & Setup Pipeline", type="primary", key='p1_estimate_button'):
            st.session_state.pipeline_stage_p1 = 'setup' # Reset stage

            if not st.session_state.api_key:
                st.error("Please provide an API Key.")
            # --- MODIFIED: Check both dropdowns ---
            elif target_company_select == company_options[0]:
                st.error("Please select a company.")
            elif not available_products or target_product_input == available_products[0]:
                st.error("Please select a product.")
            # --- END MODIFICATION ---
            else:
                try:
                    vectorstore = load_vectorstore(st.session_state.api_key)
                    if vectorstore is None:
                        st.session_state.pipeline_stage_p1 = 'setup'
                    else:
                        # MODIFIED: Pass selected model to cost estimation
                        total_tokens, total_cost, N_c_estimate = estimate_pipeline_cost_p1(
                            st.session_state.api_key,
                            merged_data,
                            target_product_input, # <-- Pass the selected product
                            num_angles,
                            st.session_state.selected_model
                        )
                        st.session_state.estimated_cost_data_p1 = {
                            'tokens': total_tokens,
                            'cost': total_cost,
                            'N_c': N_c_estimate,
                            'target': target_product_input, # <-- Store the product
                            'angles': num_angles,
                            'model_name': st.session_state.selected_model # Store model used for estimate
                        }
                        st.session_state.pipeline_stage_p1 = 'estimated'
                        st.success("Cost estimate ready for review.")
                except Exception as e:
                    st.error(f"Setup Error: {e}")
                    st.session_state.pipeline_stage_p1 = 'setup'
            st.rerun()


    # 1. Cost Estimation and Confirmation Stage
    if st.session_state.pipeline_stage_p1 == 'estimated':
        cost_data = st.session_state.estimated_cost_data_p1
        model_name = cost_data['model_name']
        model_display_name = MODEL_CONFIG[model_name]['display_name']

        st.header("💰 Cost Confirmation Required")
        st.warning(f"Target: **{cost_data['target']}** | Angles: **{cost_data['angles']}**")
        st.markdown("---")

        col1, col2 = st.columns(2)
        # MODIFIED: Display model name in metric
        col1.metric(f"Estimated Cost ({model_display_name})", f"${cost_data['cost']:.4f} USD")
        col2.metric("Estimated Total Tokens", f"{cost_data['tokens']:,}")

        st.info(f"The estimate for scoring is based on a conservative **{cost_data['N_c']}** potential candidates. The actual cost may be lower if fewer candidates are retrieved.")

        if st.button("CONFIRM & RUN FULL PIPELINE", type="secondary", key='p1_confirm_run_button'):
            st.session_state.pipeline_stage_p1 = 'running'
            st.rerun()


    # 2. Pipeline Execution Stage (Visible only when running)
    if st.session_state.pipeline_stage_p1 == 'running':
        st.header("🚀 Running RAG Pipeline...")
        st.markdown("---")

        target_product_input = st.session_state.estimated_cost_data_p1['target']
        num_angles = st.session_state.estimated_cost_data_p1['angles']
        # MODIFIED: Get model name from the stored estimate data
        model_name_to_run = st.session_state.estimated_cost_data_p1['model_name']

        # Wrap execution in try/except block
        try:
            vectorstore = load_vectorstore(st.session_state.api_key)

            # MODIFIED: Pass model name to execution function
            # This is the ORIGINAL product-to-product execution function
            df_final, df_intermediate, synergy_strategies = run_pipeline_execution_p1(
                st.session_state.api_key, merged_data, target_product_input, num_angles, vectorstore, model_name_to_run
            )

            st.session_state.final_results_p1 = {
                'target': target_product_input,
                'shortlist_df': df_final,
                'intermediate_df': df_intermediate,
                'synergy_strategies': synergy_strategies
            }

            # Set stage to complete AFTER outputting intermediate results
            st.session_state.pipeline_stage_p1 = 'complete'
            st.success(f"Pipeline 1 Complete. Found {len(df_final)} candidates.")
            st.balloons()
            st.rerun() # Rerun to switch to the final results display

        except Exception as e:
            st.error(f"❌ An unexpected error occurred during the LLM analysis: {e}")
            st.session_state.pipeline_stage_p1 = 'setup'
            st.session_state.estimated_cost_data_p1 = None
            st.exception(e)

    # 3. Final Results Stage (Visible only when complete)
    if st.session_state.pipeline_stage_p1 == 'complete':
        results = st.session_state.final_results_p1
        st.subheader(f"✅ Pipeline 1 Complete for **{results['target']}**")
        st.markdown("---")
        df_final = results['shortlist_df']
        df_intermediate = results['intermediate_df']
        synergy_strategies = results['synergy_strategies']

        # --- Display Strategist JSON Output FIRST ---
        st.subheader("1️⃣ Dynamic Strategy Generation (LLM 1)")
        st.json(synergy_strategies)
        st.markdown("---")

        if not df_final.empty:

            # --- Inline CSS for DataFrame Justification Wrapping (Applies to final table) ---
            st.markdown(
                """
                <style>
                /* Target the Justification column cells in the final DataFrame */
                .stDataFrame .data-row > div:nth-child(5) div {
                    white-space: pre-wrap !important;
                    word-wrap: break-word !important;
                    overflow-x: hidden !important;
                    max-height: 100%;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            # ----------------------------------------------------

            # Display Intermediate Candidates second
            st.subheader("2️⃣ Candidate Profiling and RAG Retrieval (LLM 2 + Chroma)")
            st.info("These candidates were retrieved by the RAG step and ranked purely on the volume of matching chunks.")
            st.dataframe(df_intermediate, use_container_width=True, hide_index=True)
            st.markdown("---")

            # Display Final Shortlist
            st.subheader("3️⃣ Final Ranked Synergy Shortlist (Ranked by LLM Score)")
            st.info("The final ranking after LLM-powered initial synergy assessment (LLM 3).")
            st.dataframe(df_final.style.bar(subset=['Score'], color='#5fb2b7'), use_container_width=True, hide_index=True)

            st.info("The top-ranked candidates are now available as a reference in Pipeline 2 for 1-on-1 deep dive analysis.")

            if st.button("Proceed to 1-on-1 Deep Dive (Pipeline 2) with Results", type="secondary"):
                st.session_state.page = 'p2'
                st.rerun()
        else:
            st.warning("No candidates were found by Pipeline 1.")


# ==============================================================================
# VIEW 2: PIPELINE 2 - 1-ON-1 DEEP DIVE ANALYSIS
# ==============================================================================
elif st.session_state.page == 'p2':
    st.header("🔎 Pipeline 2: 1-on-1 Deep Dive Synergy Analysis")
    st.markdown("This stage performs a comprehensive LLM analysis on **two selected products**.")

    # --- Sidebar Configuration for P2 ---
    with st.sidebar:
        st.header("Pipeline 2 Inputs")

        def toggle_web_data_p2():
            collected_data = st.session_state.get('collected_data_p2')
            if collected_data is None: return

            current_mode_is_web = collected_data.get('mode') == 'Web Data ON'
            new_mode_is_web = st.session_state.web_data_toggle_p2

            if current_mode_is_web != new_mode_is_web:
                 st.session_state.collected_data_p2 = None
                 st.session_state.analysis_ready_p2 = False
                 st.session_state.analysis_report_p2 = None

        st.checkbox(
            "Include External Web Scraping",
            value=st.session_state.web_data_toggle_p2,
            key='web_data_toggle_p2',
            on_change=toggle_web_data_p2
        )

        # --- MODIFIED: Dependent Dropdowns for Product 1 ---
        company_1_select = st.selectbox(
            "Product 1 Company",
            company_options,
            key='p2_company_1_select',
            index=0
        )
        available_products_1 = company_to_products_map.get(company_1_select, [])
        product_1_input = st.selectbox(
            "Product 1 Name",
            available_products_1,
            key='product_1_select_p2', # <-- Use original key
            index=0
        )
        st.sidebar.markdown("---") # Visual separator

        # --- MODIFIED: Dependent Dropdowns for Product 2 ---
        company_2_select = st.selectbox(
            "Product 2 Company",
            company_options,
            key='p2_company_2_select',
            index=0
        )
        available_products_2 = company_to_products_map.get(company_2_select, [])
        product_2_input = st.selectbox(
            "Product 2 Name",
            available_products_2,
            key='product_2_select_p2', # <-- Use original key
            index=0
        )
        # --- END MODIFICATION ---

        data_source_label = "Internal Data + Web" if st.session_state.web_data_toggle_p2 else "Internal Data Only"
        run_collection_button = st.button(f"Prepare Data ({data_source_label})", type="primary", key='p2_prepare_button')


    # --- Display P1 Results as Reference (FIX: Use markdown bullet list) ---
    if st.session_state.final_results_p1 and not st.session_state.final_results_p1['shortlist_df'].empty:
        df_p1 = st.session_state.final_results_p1['shortlist_df']
        target_p1 = st.session_state.final_results_p1['target']

        with st.expander(f"⭐ **Pipeline 1 Synergy Filtering Reference** (Target: {target_p1})", expanded=True):
            st.info("The candidates below were found by the synergy filtering pipeline (P1).")

            shortlist_markdown = ""
            for index, row in df_p1.iterrows():
                shortlist_markdown += f"* **Rank {row['Rank']} (Score {row['Score']}):** {row['Product']} by {row['Company']}\n  - *Justification:* {row['Justification']}\n"

            st.markdown(shortlist_markdown)
    else:
        st.info("No results found from Pipeline 1. Run Pipeline 1 first to generate a candidate list for reference. Or select any 2 Products to run 1-on-1 synergy analysis.")

    st.markdown("---")

    # --- Editable Prompt Section (P2) ---
    def recalculate_p2_cost_and_prompt():
        """Handles the logic for the 'Apply & Recalculate Prompt Cost' button."""
        if st.session_state.collected_data_p2 is None:
            st.warning("Please press the 'Prepare Data' button first to collect data and establish the base prompt context.")
            return
        if not st.session_state.api_key:
             st.error("Please enter your OpenAI API Key first.")
             return

        # 1. Get current data for cost estimation
        doc1 = st.session_state.collected_data_p2['product_1_doc']
        doc2 = st.session_state.collected_data_p2['product_2_doc']
        model_name = st.session_state.selected_model # Use globally selected model

        # 2. Get the current toggle state and labels (must match how data was prepared)
        use_web_data = st.session_state.web_data_toggle_p2
        data_source_label = "Internal Data + External Web Data" if use_web_data else "Internal Data ONLY"
        data_source_prompt_text = "internal data AND external web data" if use_web_data else "internal data"

        # 3. Re-assemble the final prompt with the current edited instructions
        current_instructions = st.session_state.editable_prompt_p2

        final_prompt_template = current_instructions.replace(
            "read and internalize the data for both companies",
            f"read and internalize the **{data_source_prompt_text}** for both companies"
        ).replace(
            "provided data.",
            f"provided **{data_source_prompt_text}**."
        )
        final_prompt_template += FIXED_PLACEHOLDERS.replace("{data_source_label}", data_source_label)

        # 4. Recalculate cost using the FULL final_prompt_template string and documents
        input_tokens, output_tokens, total_cost = calculate_llm_cost_p2(
            final_prompt_template, doc1, doc2, model_name
        )

        # 5. Update session state with new cost and prompt
        st.session_state.collected_data_p2['cost'] = {'input': input_tokens, 'output': output_tokens, 'total': total_cost}
        st.session_state.collected_data_p2['final_prompt_template'] = final_prompt_template
        st.session_state.collected_data_p2['model_name'] = model_name # Store model name

        st.success("Prompt applied and cost recalculated!")
        st.rerun()

    with st.expander("🛠️ View/Edit LLM Prompt Template", expanded=False):
        st.session_state.editable_prompt_p2 = st.text_area(
            "Edit Analyst Chain Prompt Template (Instructions Only)",
            value=st.session_state.editable_prompt_p2,
            height=600,
            key='prompt_text_area_p2',
        )
        # FIX: Include the 'Apply & Recalculate Prompt Cost' button for P2
        st.button("Apply & Recalculate Prompt Cost", type="secondary", on_click=recalculate_p2_cost_and_prompt)


    # --- Data Collection Logic (P2) ---
    if run_collection_button:
        st.session_state.analysis_report_p2 = None
        st.session_state.analysis_ready_p2 = False
        st.session_state.collected_data_p2 = None

        use_web_data = st.session_state.web_data_toggle_p2
        model_name = st.session_state.selected_model # MODIFIED: Get selected model

        # --- MODIFIED: Check all new dropdowns ---
        if company_1_select == company_options[0] or (not available_products_1 or product_1_input == available_products_1[0]):
            st.error("Please select a valid product for Product 1.")
        elif company_2_select == company_options[0] or (not available_products_2 or product_2_input == available_products_2[0]):
            st.error("Please select a valid product for Product 2.")
        # --- END MODIFICATION ---
        elif not st.session_state.api_key:
            st.error("Please enter your OpenAI API Key.")
        else:
            with st.spinner(f'Preparing data... (Web scraping {"ON" if use_web_data else "OFF"})...'):
                try:
                    # This section already correctly uses 'Product'
                    product_1_row = merged_data.loc[merged_data['Product'] == product_1_input].iloc[0]
                    product_2_row = merged_data.loc[merged_data['Product'] == product_2_input].iloc[0]

                    product_1_internal = product_1_row['doc_text']
                    product_2_internal = product_2_row['doc_text']
                    product_1_url, product_2_url = "", ""
                    data_source_label = "Internal Data ONLY"
                    data_source_prompt_text = "internal data"

                    if use_web_data:
                        product_1_url = extract_website_url(product_1_internal)
                        product_2_url = extract_website_url(product_2_internal)

                        product_1_external = get_external_data_snippets(product_1_url)
                        product_2_external = get_external_data_snippets(product_2_url)

                        data_source_label = "Internal Data + External Web Data"
                        data_source_prompt_text = "internal data AND external web data"

                        PRODUCT_1_DOC = (
                            f"--- PRODUCT 1: {product_1_row['Product']} by {product_1_row['Company']} ---\n"
                            f"*** Internal Data ***\n{product_1_internal}\n\n"
                            f"*** External Web Data ***\n{product_1_external}\n"
                        )
                        PRODUCT_2_DOC = (
                            f"--- PRODUCT 2: {product_2_row['Product']} by {product_2_row['Company']} ---\n"
                            f"*** Internal Data ***\n{product_2_internal}\n\n"
                            f"*** External Web Data ***\n{product_2_external}\n"
                        )

                    else:
                        PRODUCT_1_DOC = f"--- PRODUCT 1: {product_1_row['Product']} by {product_1_row['Company']} ---\n{product_1_internal}\n"
                        PRODUCT_2_DOC = f"--- PRODUCT 2: {product_2_row['Product']} by {product_2_row['Company']} ---\n{product_2_internal}\n"

                    current_instructions = st.session_state.editable_prompt_p2

                    final_prompt_template = current_instructions.replace(
                        "read and internalize the data for both companies",
                        f"read and internalize the **{data_source_prompt_text}** for both companies"
                    ).replace(
                        "provided data.",
                        f"provided **{data_source_prompt_text}**."
                    )
                    final_prompt_template += FIXED_PLACEHOLDERS.replace("{data_source_label}", data_source_label)

                    # MODIFIED: Pass model_name to cost calculation
                    input_tokens, output_tokens, total_cost = calculate_llm_cost_p2(
                        final_prompt_template, PRODUCT_1_DOC, PRODUCT_2_DOC, model_name
                    )

                    st.session_state.collected_data_p2 = {
                        'product_1_doc': PRODUCT_1_DOC,
                        'product_2_doc': PRODUCT_2_DOC,
                        'api_key': st.session_state.api_key,
                        'cost': {'input': input_tokens, 'output': output_tokens, 'total': total_cost},
                        'product_1_url': product_1_url,
                        'product_2_url': product_2_url,
                        'mode': 'Web Data ON' if use_web_data else 'Internal Data Only',
                        'final_prompt_template': final_prompt_template,
                        'model_name': model_name # Store the model name with the prepared data
                    }
                    st.session_state.analysis_ready_p2 = True
                    st.success(f"Data prepared ({st.session_state.collected_data_p2['mode']}). Ready for analysis.")

                except IndexError:
                    st.error(f"❌ Error: Product data could not be found internally.")
                    st.session_state.analysis_ready_p2 = False
                except Exception as e:
                    st.error(f"An unexpected error occurred during data processing: {e}")
                    st.exception(e)
                    st.session_state.analysis_ready_p2 = False
            st.rerun()

    # --- Manual Review and LLM Analysis Section (P2) ---
    if st.session_state.analysis_ready_p2 and st.session_state.collected_data_p2:
        data = st.session_state.collected_data_p2
        cost_data = data['cost']
        mode = data['mode']
        was_web_data_used = mode == 'Web Data ON'
        model_name = data['model_name'] # Get model name from prepared data
        model_display_name = MODEL_CONFIG[model_name]['display_name']


        st.header(f"🕵️ Manual Data Review and Cost Confirmation ({mode})")
        st.markdown("---")

        if was_web_data_used:
            st.warning("⚠️ **Mode**: Web scraping was ON. Note that external data increases input tokens and LLM cost.")

        col1, col2 = st.columns(2)

        # Get product names from the selectbox keys
        product_1_name = st.session_state.get('product_1_select_p2', 'Product 1')
        product_2_name = st.session_state.get('product_2_select_p2', 'Product 2')

        internal_data_1, external_data_1 = get_data_for_display(data['product_1_doc'], was_web_data_used)
        internal_data_2, external_data_2 = get_data_for_display(data['product_2_doc'], was_web_data_used)

        with col1:
            st.subheader(f"Product 1 ({product_1_name}) Data")
            st.text_area("Product 1 Internal Data", internal_data_1, height=150)
            if was_web_data_used:
                st.caption(f"Source URL: {data.get('product_1_url') or 'N/A'}")
                st.text_area("Product 1 External Data", external_data_1, height=150)

        with col2:
            st.subheader(f"Product 2 ({product_2_name}) Data")
            st.text_area("Product 2 Internal Data", internal_data_2, height=150)
            if was_web_data_used:
                st.caption(f"Source URL: {data.get('product_2_url') or 'N/A'}")
                st.text_area("Product 2 External Data", external_data_2, height=150)


        st.markdown("---")
        # MODIFIED: Show model name in header
        st.subheader(f"💰 LLM Cost Estimate ({model_display_name})")
        st.warning(f"Estimated Total Cost for ONE analysis run: **${cost_data['total']:.4f} USD**")

        cost_col1, cost_col2 = st.columns(2)
        cost_col1.metric("Input Tokens (Est.)", f"{cost_data['input']:,}")
        cost_col2.metric("Output Tokens (Est.)", f"{cost_data['output']:,}")

        if st.button("Confirm & Run Final LLM Analysis", type="secondary", key='p2_run_final'):
            with st.spinner("🧠 Running 1-on-1 Deep Dive Analyst Chain..."):
                try:
                    # MODIFIED: Pass model name to chain definition
                    analyst_chain = define_analyst_chain_p2(data['api_key'], data['final_prompt_template'], model_name)

                    analysis_report = analyst_chain.invoke({
                        "product_1_doc": data['product_1_doc'],
                        "product_2_doc": data['product_2_doc']
                    })

                    st.session_state.analysis_report_p2 = analysis_report
                    st.success("Analysis Complete!")
                    st.balloons()
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ An error occurred during the LLM analysis: {e}. Check your API key and network connection.")
                    st.exception(e)

    # --- Analysis Report Display (P2) ---
    if st.session_state.analysis_report_p2:
        st.subheader("📈 Executive Partnership Briefing")
        st.markdown("---")

        st.markdown(
            """
            <style>
            .output-box-readable {
                white-space: pre-wrap !important;
                word-wrap: break-word !important;
                overflow-x: hidden !important;
                font-family: monospace;
                background-color: #262730;
                color: #FAFAFA;
                border-radius: 0.3rem;
                padding: 1rem;
                border: 1px solid #333;
                max-width: 100%;
            }
            .output-box-readable strong {
                color: #FCF4A3;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        output_content = st.session_state.analysis_report_p2.replace('\n', '<br>')
        st.markdown(
            f'<div class="output-box-readable">{output_content}</div>',
            unsafe_allow_html=True
        )