
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import plotly.express as px
import warnings
import logging
import os
from pathlib import Path

# Suppress HuggingFace / transformers warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

st.set_page_config(page_title="Product Compliance Checker", page_icon="🔍", layout="wide")

st.markdown("""
    <style>
    .header-main { font-size: 2.5rem; color: #0066cc; font-weight: bold; }
    .decision-compliant { color: #28a745; font-size: 1.2rem; font-weight: bold; }
    .decision-non { color: #dc3545; font-size: 1.2rem; font-weight: bold; }
    .decision-review { color: #ffc107; font-size: 1.2rem; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# LOAD MODELS


@st.cache_resource
def load_data_assets():
    try:
        base_dir = Path(__file__).resolve().parent
        artifact_dir = base_dir / "app_artifacts"

        df = pd.read_csv(artifact_dir / "Data.csv")
        with open(artifact_dir / "embeddings.pkl", "rb") as f:
            embeddings = pickle.load(f)
        faiss_index = faiss.read_index(str(artifact_dir / "faiss_index.bin"))
        with open(artifact_dir / "model_config.pkl", "rb") as f:
            model_config = pickle.load(f)

        
        local_embedding_path = artifact_dir / "embedding_model"
        local_llm_path = artifact_dir / "llm_qwen"

        return {
            "df": df,
            "embeddings": embeddings,
            "faiss_index": faiss_index,
            "embedding_model_name": model_config.get("embedding_model_name", "all-MiniLM-L6-v2"),
            "embedding_model_path": str(local_embedding_path) if local_embedding_path.exists() else None,
            "llm_model_path": str(local_llm_path) if local_llm_path.exists() else None,
        }
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.info("Required inside app_artifacts: Data.csv | embeddings.pkl | faiss_index.bin | model_config.pkl | embedding_model/ | llm_qwen/")
        return None

@st.cache_resource
def load_embedding_model(model_name_or_path):
    """Load the embedding model only when semantic search is requested."""
    return SentenceTransformer(model_name_or_path)


@st.cache_resource
def load_local_llm(llm_path):
    """Load a locally saved tokenizer and LLM only when explanation polishing is needed."""
    tokenizer = AutoTokenizer.from_pretrained(llm_path, local_files_only=True)
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        local_files_only=True,
    )
    return tokenizer, llm_model


# CLEAN DATA

@st.cache_data
def clean_data(df):
    df = df.copy()
    columns_to_clean = [
        'hazard_type', 'has_safety_warning', 'safety_warning_text',
        'compliance_certifications', 'regulatory_body', 'compliance_status'
    ]
    
    for col in columns_to_clean:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].replace(['nan', 'NaN', 'None', 'none', '', 'NAN', 'na', 'NA', 'N/A', 'n/a'], 'Not Provided')
            df[col] = df[col].fillna('Not Provided')
    
    return df

# RETRIEVE & ANALYZE

def retrieve_similar_records(models, query, top_k=5):
    embedding_source = models.get("embedding_model_path") or models["embedding_model_name"]
    embedding_model = load_embedding_model(embedding_source)
    faiss_index = models["faiss_index"]
    df = models["df"]
    
    query_embedding = embedding_model.encode(
        [str(query)],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")
    
    similarity_scores, retrieved_indices = faiss_index.search(query_embedding, top_k)
    results = df.iloc[retrieved_indices[0]].copy()
    results["similarity_score"] = similarity_scores[0]
    return results

def analyze_retrieved_evidence(retrieved_df):
    top_similarity = float(retrieved_df["similarity_score"].iloc[0])
    
    if top_similarity < 0.35:
        return {
            "decision": "Needs Review",
            "reason": "Data not provided - Product was not found clearly in the dataset.",
            "suggestion": "Enter a more specific product name or manually verify certification, hazard type, and regulatory body."
        }
    
    relevant_df = retrieved_df[retrieved_df["similarity_score"] >= max(0.35, top_similarity - 0.08)].copy()
    
    relevant_df["status_clean"] = relevant_df["compliance_status"].astype(str).str.strip().str.lower()
    relevant_df["hazard_clean"] = relevant_df["hazard_type"].astype(str).str.strip().str.lower()
    relevant_df["warning_clean"] = relevant_df["has_safety_warning"].astype(str).str.strip().str.lower()
    
    compliant_count = (relevant_df["status_clean"] == "compliant").sum()
    non_compliant_count = (relevant_df["status_clean"] == "non-compliant").sum()
    pending_count = (relevant_df["status_clean"] == "pending").sum()
    
    serious_warning_issue = (
        relevant_df["hazard_clean"].isin(["chemical", "electrical"]) &
        (relevant_df["warning_clean"] != "yes")
    ).sum()
    
    if non_compliant_count >= 2:
        return {
            "decision": "Non-Compliant",
            "reason": "Data provided - Multiple relevant records are marked non-compliant.",
            "suggestion": "Review certification, safety warning, and regulatory approval before sale."
        }
    
    if serious_warning_issue >= 2:
        return {
            "decision": "Non-Compliant",
            "reason": "Data provided - Relevant records show hazards without required safety warnings.",
            "suggestion": "Add proper safety warnings and verify label compliance."
        }
    
    if non_compliant_count == 1 or pending_count > 0:
        return {
            "decision": "Needs Review",
            "reason": "Data provided - Product details may include mixed or pending compliance evidence.",
            "suggestion": "Verify the exact product batch, certification, and regulatory approval before final approval."
        }
    
    if compliant_count >= 1 and serious_warning_issue == 0:
        return {
            "decision": "Compliant",
            "reason": "Data provided - The relevant retrieved record is compliant with valid certification and required warning evidence.",
            "suggestion": "Maintain updated certification and regulatory documentation."
        }
    
    return {
        "decision": "Needs Review",
        "reason": "Data not provided - Retrieved evidence is not strong enough for a confident compliance decision.",
        "suggestion": "Verify the exact product certification, hazard type, and regulatory body before approval."
    }


def generate_llm_response(models, prompt, max_new_tokens=120):
    llm_path = models.get("llm_model_path")
    if not llm_path or not os.path.exists(llm_path):
        return None
    tokenizer, llm_model = load_local_llm(llm_path)
    messages = [
        {"role": "system", "content": "You are a product compliance assistant. Keep answers short and structured."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(llm_model.device)
    with torch.no_grad():
        output_ids = llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def polish_evidence_with_llm(models, product_name, evidence):
    prompt = f"""
The compliance decision is already fixed. Do not change it.

Product query: {product_name}
Compliance Decision: {evidence['decision']}
Evidence Reason: {evidence['reason']}
Evidence Suggestion: {evidence['suggestion']}

Rewrite only the reason and suggestion in simple, professional language.
Return exactly two lines:
Reason: one short sentence
Suggestion: one short sentence
""".strip()
    try:
        raw = generate_llm_response(models, prompt)
        if not raw:
            return evidence
        reason = ""
        suggestion = ""
        for line in raw.splitlines():
            clean = line.strip()
            lower = clean.lower()
            if lower.startswith("reason:"):
                reason = clean.split(":", 1)[1].strip()
            elif lower.startswith("suggestion:"):
                suggestion = clean.split(":", 1)[1].strip()
        polished = evidence.copy()
        if reason:
            polished["reason"] = reason
        if suggestion:
            polished["suggestion"] = suggestion
        return polished
    except Exception:
        return evidence

def check_product_compliance(models, product_name):
    product_name = str(product_name).strip()
    
    if product_name == "":
        return {
            "decision": "Needs Review",
            "reason": "Data not provided - No product name was entered.",
            "suggestion": "Enter a valid product name for compliance checking."
        }, None
    
    retrieved_df = retrieve_similar_records(models, product_name, top_k=5)
    evidence = analyze_retrieved_evidence(retrieved_df)
    return evidence, retrieved_df

def normalize_text(value):
    return str(value).strip() if pd.notna(value) else "Not Provided"


def evaluate_custom_compliance(
    hazard,
    safety_warning,
    age_restriction,
    contains_hazardous_substances,
    certification,
    regulatory_body,
):
    """Rule-based live decision using manually editable product attributes."""
    hazard = normalize_text(hazard)
    safety_warning = normalize_text(safety_warning)
    age_restriction = normalize_text(age_restriction)
    contains_hazardous_substances = normalize_text(contains_hazardous_substances)
    certification = normalize_text(certification)
    regulatory_body = normalize_text(regulatory_body)

    hazard_l = hazard.lower()
    safety_warning_l = safety_warning.lower()
    age_restriction_l = age_restriction.lower()
    hazardous_substances_l = contains_hazardous_substances.lower()
    certification_l = certification.lower()
    regulatory_body_l = regulatory_body.lower()

    missing_certification = certification_l in ["not provided", "none", ""]
    missing_regulator = regulatory_body_l in ["not provided", "none", ""]
    has_hazardous_substances = hazardous_substances_l == "yes"
    dangerous_hazard = hazard_l in ["chemical", "electrical", "physical"]
    low_age_band = age_restriction_l in ["all ages", "3+", "5+"]

    if dangerous_hazard and safety_warning_l in ["no", "not provided"]:
        return {
            "decision": "Non-Compliant",
            "reason": f"Dangerous hazard ({hazard}) without a clear safety warning.",
            "suggestion": "Add appropriate safety warnings before approval."
        }
    if hazard_l in ["chemical", "electrical"] and low_age_band:
        return {
            "decision": "Non-Compliant",
            "reason": f"High-risk hazard ({hazard}) is not suitable for low age restriction ({age_restriction}).",
            "suggestion": "Use a safer design or increase the age restriction."
        }
    if has_hazardous_substances and safety_warning_l != "yes":
        return {
            "decision": "Non-Compliant",
            "reason": "Hazardous substances are present but warning coverage is missing.",
            "suggestion": "Provide clear warnings and verify the product label."
        }
    if missing_certification or missing_regulator:
        missing_items = []
        if missing_certification:
            missing_items.append("certification")
        if missing_regulator:
            missing_items.append("regulatory body")
        return {
            "decision": "Needs Review",
            "reason": f"Required compliance evidence is incomplete: {', '.join(missing_items)} missing.",
            "suggestion": "Add the missing regulatory evidence before final approval."
        }
    if dangerous_hazard or has_hazardous_substances:
        return {
            "decision": "Needs Review",
            "reason": "Risk-related attributes are present, but required controls are documented.",
            "suggestion": "Confirm the certification and warning text against the exact regulation."
        }
    return {
        "decision": "Compliant",
        "reason": "No material risk flags found and required regulatory evidence is present.",
        "suggestion": "Maintain the supporting documents and keep them up to date."
    }


def add_row_predictions(evidence_df):
    evidence_df = evidence_df.copy()
    evidence_df["predicted_decision"] = evidence_df.apply(
        lambda row: evaluate_custom_compliance(
            row.get("hazard_type", "Not Provided"),
            row.get("has_safety_warning", "Not Provided"),
            row.get("age_restriction", "Not Provided"),
            row.get("contains_hazardous_substances", "Not Provided"),
            row.get("compliance_certifications", "Not Provided"),
            row.get("regulatory_body", "Not Provided"),
        )["decision"],
        axis=1,
    )
    return evidence_df


def analyze_edited_evidence(evidence_df):
    predicted = evidence_df["predicted_decision"].astype(str)
    non_compliant_count = (predicted == "Non-Compliant").sum()
    review_count = (predicted == "Needs Review").sum()
    compliant_count = (predicted == "Compliant").sum()

    if non_compliant_count >= 2:
        return {
            "decision": "Non-Compliant",
            "reason": f"{non_compliant_count} edited similar records are predicted non-compliant.",
            "suggestion": "Resolve the non-compliant rows before approval."
        }
    if non_compliant_count == 1 or review_count > 0:
        return {
            "decision": "Needs Review",
            "reason": f"Edited evidence contains {non_compliant_count} non-compliant and {review_count} review-needed row(s).",
            "suggestion": "Review the flagged rows or update missing evidence before final approval."
        }
    if compliant_count == len(evidence_df) and len(evidence_df) > 0:
        return {
            "decision": "Compliant",
            "reason": "All edited similar records are currently predicted compliant.",
            "suggestion": "Maintain the supporting evidence and documentation."
        }
    return {
        "decision": "Needs Review",
        "reason": "Edited evidence is not strong enough for a confident final decision.",
        "suggestion": "Review the manually entered values and supporting documents."
    }


def build_attribute_comparison(original_values, edited_values):
    """Return a dataframe-friendly comparison of original versus edited attributes."""
    rows = []
    for label, original_value in original_values.items():
        edited_value = edited_values[label]
        rows.append({
            "Attribute": label,
            "Original": original_value,
            "Edited": edited_value,
            "Changed": "Yes" if original_value != edited_value else "No",
        })
    return pd.DataFrame(rows)


def summarize_attribute_changes(original_values, edited_values):
    changes = []
    for label, original_value in original_values.items():
        edited_value = edited_values[label]
        if original_value != edited_value:
            changes.append(f"{label}: {original_value} -> {edited_value}")
    return changes


def recommend_compliance_fixes(
    hazard,
    safety_warning,
    age_restriction,
    contains_hazardous_substances,
    certification,
    regulatory_body,
):
    """Recommend the smallest practical edits that would improve the decision."""
    recommendations = []

    if hazard in ["Chemical", "Electrical", "Physical"] and safety_warning in ["No", "Not Provided"]:
        recommendations.append("Set **Safety Warning Available** to **Yes**.")
    if hazard in ["Chemical", "Electrical"] and age_restriction in ["All ages", "3+", "5+"]:
        recommendations.append("Increase **Age Restriction** to **12+** or **18+**.")
    if contains_hazardous_substances == "Yes" and safety_warning != "Yes":
        recommendations.append("Add a safety warning because hazardous substances are present.")
    if certification in ["Not Provided", "None", ""]:
        recommendations.append("Provide a valid **Compliance Certification**.")
    if regulatory_body in ["Not Provided", "None", ""]:
        recommendations.append("Provide the approving **Regulatory Body**.")

    if not recommendations and (hazard in ["Chemical", "Electrical", "Physical"] or contains_hazardous_substances == "Yes"):
        recommendations.append("Keep the current controls, then manually verify the warning text against the applicable regulation.")
    if not recommendations:
        recommendations.append("No urgent fix is needed from the edited attributes; keep documentation current.")

    return recommendations


def get_product_edit_options(df):
    return {
        "category": sorted(df["category"].dropna().astype(str).unique().tolist()),
        "brand_name": sorted(df["brand_name"].dropna().astype(str).unique().tolist()),
        "hazard_type": sorted(set(["None", "Chemical", "Electrical", "Physical", "Not Provided"] + df["hazard_type"].dropna().astype(str).tolist())),
        "has_safety_warning": sorted(set(["Yes", "No", "Not Provided"] + df["has_safety_warning"].dropna().astype(str).tolist())),
        "age_restriction": sorted(set(["All ages", "3+", "5+", "12+", "18+", "Not Provided"] + df["age_restriction"].dropna().astype(str).tolist())),
        "contains_hazardous_substances": sorted(set(["Yes", "No", "Not Provided"] + df["contains_hazardous_substances"].dropna().astype(str).tolist())),
        "compliance_certifications": sorted(set(["Not Provided"] + df["compliance_certifications"].dropna().astype(str).tolist())),
        "regulatory_body": sorted(set(["Not Provided"] + df["regulatory_body"].dropna().astype(str).tolist())),
    }


def render_product_details(product):
    fields = [
        ("Product Name", "product_name"),
        ("Brand", "brand_name"),
        ("Category", "category"),
        ("Description", "description"),
        ("Hazard Type", "hazard_type"),
        ("Safety Warning Available", "has_safety_warning"),
        ("Safety Warning Text", "safety_warning_text"),
        ("Age Restriction", "age_restriction"),
        ("Contains Hazardous Substances", "contains_hazardous_substances"),
        ("Compliance Certification", "compliance_certifications"),
        ("Regulatory Body", "regulatory_body"),
        ("Compliance Status", "compliance_status"),
        ("Usage Instructions", "usage_instructions"),
        ("Storage Instructions", "storage_instructions"),
        ("Country of Origin", "country_of_origin"),
        ("SKU", "sku"),
        ("Barcode", "barcode"),
    ]
    rows = [{"Attribute": label, "Value": normalize_text(product.get(col, "Not Provided"))} for label, col in fields if col in product.index]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def product_to_edit_table(product):
    cols = [
        "product_name", "brand_name", "category", "description", "hazard_type",
        "has_safety_warning", "safety_warning_text", "age_restriction",
        "contains_hazardous_substances", "compliance_certifications", "regulatory_body",
    ]
    return pd.DataFrame([{col: normalize_text(product.get(col, "Not Provided")) for col in cols}])


def apply_edit_table_to_product(product, edit_df):
    updated = product.copy()
    for col in edit_df.columns:
        updated[col] = edit_df.iloc[0][col]
    return updated


def render_product_editor(product, options, key_prefix):
    edited = product.copy()
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            edited["product_name"] = st.text_input("Product Name", normalize_text(product.get("product_name")), key=f"{key_prefix}_product_name")
            edited["brand_name"] = st.selectbox("Brand", options["brand_name"], index=options["brand_name"].index(normalize_text(product.get("brand_name"))) if normalize_text(product.get("brand_name")) in options["brand_name"] else None, accept_new_options=True, key=f"{key_prefix}_brand")
            edited["category"] = st.selectbox("Category", options["category"], index=options["category"].index(normalize_text(product.get("category"))) if normalize_text(product.get("category")) in options["category"] else None, accept_new_options=True, key=f"{key_prefix}_category")
            edited["hazard_type"] = st.selectbox("Hazard Type", options["hazard_type"], index=options["hazard_type"].index(normalize_text(product.get("hazard_type"))) if normalize_text(product.get("hazard_type")) in options["hazard_type"] else None, accept_new_options=True, key=f"{key_prefix}_hazard")
            edited["has_safety_warning"] = st.selectbox("Safety Warning Available", options["has_safety_warning"], index=options["has_safety_warning"].index(normalize_text(product.get("has_safety_warning"))) if normalize_text(product.get("has_safety_warning")) in options["has_safety_warning"] else None, accept_new_options=True, key=f"{key_prefix}_warning")
        with c2:
            edited["age_restriction"] = st.selectbox("Age Restriction", options["age_restriction"], index=options["age_restriction"].index(normalize_text(product.get("age_restriction"))) if normalize_text(product.get("age_restriction")) in options["age_restriction"] else None, accept_new_options=True, key=f"{key_prefix}_age")
            edited["contains_hazardous_substances"] = st.selectbox("Hazardous Substances", options["contains_hazardous_substances"], index=options["contains_hazardous_substances"].index(normalize_text(product.get("contains_hazardous_substances"))) if normalize_text(product.get("contains_hazardous_substances")) in options["contains_hazardous_substances"] else None, accept_new_options=True, key=f"{key_prefix}_hazardous")
            edited["compliance_certifications"] = st.selectbox("Compliance Certification", options["compliance_certifications"], index=options["compliance_certifications"].index(normalize_text(product.get("compliance_certifications"))) if normalize_text(product.get("compliance_certifications")) in options["compliance_certifications"] else None, accept_new_options=True, key=f"{key_prefix}_cert")
            edited["regulatory_body"] = st.selectbox("Regulatory Body", options["regulatory_body"], index=options["regulatory_body"].index(normalize_text(product.get("regulatory_body"))) if normalize_text(product.get("regulatory_body")) in options["regulatory_body"] else None, accept_new_options=True, key=f"{key_prefix}_regulator")
            edited["description"] = st.text_area("Description", normalize_text(product.get("description")), key=f"{key_prefix}_description")
        edited["safety_warning_text"] = st.text_area("Safety Warning Text", normalize_text(product.get("safety_warning_text")), key=f"{key_prefix}_warning_text")
    return edited


def product_rule_decision(product):
    return evaluate_custom_compliance(
        product.get("hazard_type", "Not Provided"),
        product.get("has_safety_warning", "Not Provided"),
        product.get("age_restriction", "Not Provided"),
        product.get("contains_hazardous_substances", "Not Provided"),
        product.get("compliance_certifications", "Not Provided"),
        product.get("regulatory_body", "Not Provided"),
    )

# MAIN APP

def main():
    st.markdown('<h1 class="header-main"> Product Compliance Checker</h1>', unsafe_allow_html=True)
    st.markdown("Filter by Category, Certification & Compliance Status | Search & Analyze Products")
    st.divider()
    
    with st.spinner("Loading data..."):
        models = load_data_assets()
    
    if models is None:
        st.stop()
    
    df_clean = clean_data(models["df"])

    categories = sorted(df_clean['category'].unique().tolist())
    certifications = sorted([
        x for x in df_clean['compliance_certifications'].unique().tolist()
        if x != "Not Provided"
    ])
    statuses = sorted(df_clean['compliance_status'].unique().tolist())

    active_view = st.radio(
        "Choose view",
        ["Search Compliance Analysis", "Overview & Visualization"],
        horizontal=True,
        label_visibility="collapsed",
    )

    selected_categories = []
    selected_certifications = []
    selected_statuses = []

    if active_view == "Overview & Visualization":
        with st.sidebar:
            st.markdown('<h1 class="header-main"> FILTERS </h1>', unsafe_allow_html=True)

            st.subheader("Category")
            cat_options = ["Select All"] + categories
            cat_selection = st.multiselect(
                "Select Categories",
                options=cat_options,
                default=[],
                label_visibility="collapsed",
                key="cat_multiselect"
            )
            selected_categories = categories if "Select All" in cat_selection else cat_selection

            st.divider()

            st.subheader("Certifications")
            cert_options = ["Select All"] + certifications
            cert_selection = st.multiselect(
                "Select Certifications",
                options=cert_options,
                default=[],
                label_visibility="collapsed",
                key="cert_multiselect"
            )
            selected_certifications = certifications if "Select All" in cert_selection else cert_selection

            st.divider()

            st.subheader("Compliance Status")
            status_options = ["Select All"] + statuses
            status_selection = st.multiselect(
                "Select Status",
                options=status_options,
                default=[],
                label_visibility="collapsed",
                key="status_multiselect"
            )
            selected_statuses = statuses if "Select All" in status_selection else status_selection

    # APPLY FILTERS

    cat_filter = selected_categories if selected_categories else categories
    cert_filter = selected_certifications if selected_certifications else certifications + ["Not Provided"]
    status_filter = selected_statuses if selected_statuses else statuses

    filtered_df = df_clean[
        (df_clean['category'].isin(cat_filter)) &
        (df_clean['compliance_certifications'].isin(cert_filter)) &
        (df_clean['compliance_status'].isin(status_filter))
    ].copy()

    if active_view == "Search Compliance Analysis":
        st.subheader("Product Search")
        product_options = sorted(df_clean["product_name"].dropna().astype(str).unique().tolist())
        s1, s2 = st.columns([4, 1])
        with s1:
            search_query = st.selectbox(
                "Search or choose product",
                options=product_options,
                index=None,
                placeholder="Type a product name or choose from the dropdown",
                accept_new_options=True,
                label_visibility="collapsed",
                key="product_search_query",
            )
        with s2:
            search_button = st.button("Search", type="primary", width="stretch", key="product_search_button")

        if "selected_product_index" not in st.session_state:
            st.session_state.selected_product_index = None
        if "show_all_matches" not in st.session_state:
            st.session_state.show_all_matches = False
        if "checked_main_product" not in st.session_state:
            st.session_state.checked_main_product = False
        if "editing_main_product" not in st.session_state:
            st.session_state.editing_main_product = False
        if "edited_products" not in st.session_state:
            st.session_state.edited_products = {}

        if search_query and (search_button or st.session_state.selected_product_index is None):
            match_df = df_clean[df_clean["product_name"].str.contains(str(search_query), case=False, na=False)].copy()
            match_df = match_df.drop_duplicates(subset=["product_name"]).reset_index().rename(columns={"index": "source_index"})
            st.markdown("### Matching Products")
            if match_df.empty:
                st.info("No matching product names found in the dataset.")
            else:
                visible_matches = match_df if st.session_state.show_all_matches else match_df.head(5)
                for _, row in visible_matches.iterrows():
                    if st.button(row["product_name"], key=f"match_{row['source_index']}", width="stretch"):
                        st.session_state.selected_product_index = int(row["source_index"])
                        st.session_state.checked_main_product = False
                        st.session_state.editing_main_product = False
                        st.rerun()
                if len(match_df) > 5:
                    if not st.session_state.show_all_matches:
                        if st.button(f"Show more ({len(match_df) - 5} more)", key="show_more_matches"):
                            st.session_state.show_all_matches = True
                            st.rerun()
                    elif st.button("Show fewer", key="show_fewer_matches"):
                        st.session_state.show_all_matches = False
                        st.rerun()

        selected_index = st.session_state.selected_product_index
        if selected_index is not None:
            st.divider()
            source_product = df_clean.loc[selected_index].copy()
            product = st.session_state.edited_products.get(selected_index, source_product).copy()
            st.markdown("### Selected Product Record")
            if st.session_state.editing_main_product:
                edited_product = render_product_editor(product, get_product_edit_options(df_clean), key_prefix=f"main_{selected_index}")
            else:
                render_product_details(product)

            a1, a2, a3, _ = st.columns([1, 1, 1, 3])
            with a1:
                if st.button("Check Compliance", type="primary", key="check_main_product"):
                    st.session_state.checked_main_product = True
                    st.session_state.editing_main_product = False
                    st.rerun()
            with a2:
                if st.session_state.checked_main_product and not st.session_state.editing_main_product and st.button("Edit", key="edit_main_product"):
                    st.session_state.editing_main_product = True
                    st.rerun()
            with a3:
                if st.session_state.editing_main_product and st.button("Save Changes", key="save_main_product"):
                    st.session_state.edited_products[selected_index] = edited_product
                    st.session_state.checked_main_product = False
                    st.session_state.editing_main_product = False
                    st.success("Changes saved. Click Check Compliance again to see the updated result.")
                    st.rerun()

            if st.session_state.checked_main_product:
                evidence = product_rule_decision(product)
                evidence = polish_evidence_with_llm(models, product.get("product_name", ""), evidence)
                st.markdown("### Main Product Compliance Result")
                if evidence["decision"] == "Compliant":
                    st.success(f"Compliance Decision: {evidence['decision']}")
                elif evidence["decision"] == "Non-Compliant":
                    st.error(f"Compliance Decision: {evidence['decision']}")
                else:
                    st.warning(f"Compliance Decision: {evidence['decision']}")
                c1, c2 = st.columns(2)
                with c1:
                    st.info(f"**Reason:** {evidence['reason']}")
                with c2:
                    st.warning(f"**Suggestion:** {evidence['suggestion']}")

                st.divider()
                st.markdown("### Similar Products Retrieved")
                similar_key = f"similar_edits_{selected_index}"
                if similar_key in st.session_state:
                    similar_df = st.session_state[similar_key].copy()
                else:
                    with st.spinner("Retrieving similar products..."):
                        similar_df = retrieve_similar_records(models, product.get("product_name", ""), top_k=5)
                    similar_df = add_row_predictions(similar_df)
                table_cols = ["product_name", "category", "brand_name", "hazard_type", "has_safety_warning", "age_restriction", "contains_hazardous_substances", "compliance_certifications", "regulatory_body", "similarity_score", "predicted_decision"]
                display_similar = similar_df[[c for c in table_cols if c in similar_df.columns]].copy()
                if "similarity_score" in display_similar.columns:
                    display_similar["similarity_score"] = display_similar["similarity_score"].round(4)
                st.dataframe(display_similar, width="stretch", hide_index=True)

                row_labels = [f"Row {i + 1} - {name}" for i, name in enumerate(similar_df["product_name"].astype(str).tolist())]
                chosen_row = st.selectbox("Choose a similar record to edit", row_labels, index=None, placeholder="Select a row")
                if chosen_row:
                    pos = row_labels.index(chosen_row)
                    chosen_data = similar_df.iloc[pos].copy()
                    st.markdown("### Edit Selected Similar Record")
                    edited_row = render_product_editor(chosen_data, get_product_edit_options(df_clean), key_prefix=f"similar_{selected_index}_{pos}")
                    if st.button("Save Similar Record Changes", key=f"save_similar_{selected_index}_{pos}"):
                        similar_df.iloc[pos] = edited_row
                        similar_df = add_row_predictions(similar_df)
                        st.session_state[similar_key] = similar_df
                        st.success("Similar record updated for this session.")
                        st.rerun()

    # TAB 2: OVERVIEW & VISUALIZATION 
    if active_view == "Overview & Visualization":
        st.subheader("Dashboard Overview")
        
        # Enormous Feature: Interactive Metric Cards
        total_products = len(filtered_df)
        compliant_count = len(filtered_df[filtered_df['compliance_status'].str.lower() == 'compliant'])
        non_compliant_count = len(filtered_df[filtered_df['compliance_status'].str.lower() == 'non-compliant'])
        review_count = total_products - compliant_count - non_compliant_count
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Products", f"{total_products}")
        m2.metric("Compliant", f"{compliant_count}", delta=f"{(compliant_count/total_products*100):.1f}%" if total_products else "0%")
        m3.metric("Non-Compliant", f"{non_compliant_count}", delta=f"-{(non_compliant_count/total_products*100):.1f}%" if total_products else "0%", delta_color="inverse")
        m4.metric("Needs Review / Pending", f"{review_count}")
        
        st.divider()
        
        if total_products > 0:
            col_table, col_dl = st.columns([4, 1])
            with col_table:
                st.subheader(f"Filtered Products ({total_products} results)")
            with col_dl:
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name='filtered_compliance_data.csv',
                    mime='text/csv',
                    use_container_width=True
                )
                
            display_cols = [
                'product_name', 'category', 'brand_name', 'hazard_type',
                'has_safety_warning', 'compliance_certifications',
                'compliance_status', 'regulatory_body'
            ]
            display_df = filtered_df[[col for col in display_cols if col in filtered_df.columns]].copy()
            st.dataframe(display_df, width="stretch", hide_index=True)
            st.divider()
            
            # Additional Charts
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.subheader("Compliance Status by Category")
                viz_df = filtered_df.groupby(['category', 'compliance_status']).size().reset_index(name='count')
                
                fig_bar = px.bar(
                    viz_df,
                    x='category',
                    y='count',
                    color='compliance_status',
                    barmode='group',
                    labels={'count': 'Number of Products', 'category': 'Category'},
                    color_discrete_map={
                        'Compliant': '#28a745',
                        'Non-Compliant': '#dc3545',
                        'Pending': '#ffc107',
                        'Needs Review': '#17a2b8',
                        'Not Provided': '#6c757d'
                    },
                    hover_data=['count']
                )
                fig_bar.update_layout(height=400, xaxis_tickangle=-45, showlegend=True, margin=dict(t=20))
                st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
                
            with col_chart2:
                if 'hazard_type' in filtered_df.columns:
                    st.subheader("Hazard Type Distribution")
                    pie_df = filtered_df['hazard_type'].value_counts().reset_index()
                    pie_df.columns = ['Hazard Type', 'Count']
                    fig_pie = px.pie(
                        pie_df, 
                        names='Hazard Type', 
                        values='Count',
                        hole=0.4,
                        color_discrete_sequence=px.colors.sequential.YlOrRd[::-1]
                    )
                    fig_pie.update_layout(height=400, margin=dict(t=20))
                    st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})
        else:
            st.warning(" No products found with selected filters.")

    st.divider()
    st.caption("FAISS Vector Search | Embeddings: Sentence-Transformers")

if __name__ == "__main__":
    main()
