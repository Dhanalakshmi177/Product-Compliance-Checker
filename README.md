# Product-Compliance-Checker

An AI-powered Streamlit dashboard that retrieves product compliance records using vector search and lets users interactively modify compliance attributes to get a live regulatory decision — backed by a local LLM chatbot.

# Overview

Product Compliance Checker is an end-to-end intelligent compliance analysis tool built for regulatory and quality assurance teams. A user simply searches for a product by name or category, and the system retrieves the most relevant compliance records from a dataset using semantic vector search. The user can then tweak key compliance attributes — like hazard type, safety warnings, or certifications — and the compliance decision updates in real time. An integrated AI chatbot (Qwen2.5-0.5B) answers follow-up questions based on the retrieved product records.

# Problem Statement

Manual product compliance verification is slow, error-prone, and non-interactive. Compliance officers often need to assess whether a product meets regulatory standards based on multiple attributes — certifications, hazard levels, safety warnings, age restrictions, and regulatory body approvals. Existing tools are either static lookup tables or require deep technical expertise. There is no interactive system that lets a user search a product, view evidence, simulate attribute changes, and get an instant updated compliance verdict — all in one place.

# Usage of This App

Who uses it:

Compliance officers verifying product safety before listing
Quality assurance teams doing batch compliance checks
Regulatory analysts simulating what-if attribute changes

How it works for the user:

Enter a product name (e.g., Patanjali Aloe Vera Gel) or browse by category from the sidebar
The system retrieves the top 5 most semantically similar records from the compliance dataset

# Workflow

1. User Input & Query Encoding — how the search query becomes a vector via Sentence-Transformers
2. Retrieval using FAISS — The RAG Retrieval Step — clearly explains where RAG starts, what FAISS does, and why retrieved records are the grounded evidence base
3. Live Attribute Editing & Compliance Decision Engine — explains the interactive editor and rule-based live decision logic with real examples
4. AI Chatbot — The RAG Generation Step using SLM — explains where RAG ends (generation), why Qwen2.5-0.5B is an SLM specifically, and why using a local SLM matters for privacy in regulated environments

#  Conclusion

Product Compliance Checker bridges the gap between raw compliance data and actionable regulatory decisions. By combining semantic search (FAISS + Sentence-Transformers), a live rule-based decision engine, and a locally-hosted LLM chatbot, it delivers an interactive, private, and intelligent compliance workflow — all without requiring internet access or cloud APIs. It is especially suited for teams that deal with large product catalogs and need fast, explainable compliance verdicts with the ability to simulate changes before finalizing a decision.

# Reference Links

Streamlit             https://docs.streamlit.io
FAISS                 https://github.com/facebookresearch/faiss
Sentence-Transformers   https://www.sbert.net
Hugging Face Transformers    https://huggingface.co/docs/transformers
Qwen2.5-0.5B Model         https://huggingface.co/Qwen/Qwen2.5-0.5B
PyTorch                   https://pytorch.org/docs
Pandas                    https://pandas.pydata.org/docs
RAG (Retrieval-Augmented Generation)    https://huggingface.co/blog/rag
FAISS Documentation        https://faiss.ai
Select any retrieved record and edit its attributes — hazard type, safety warning, certifications, age restriction, regulatory body, compliance status
The Compliance Decision (✅ Compliant / ❌ Non-Compliant / ⚠️ Needs Review) updates live without re-running the search
Switch to the AI Chatbot tab and ask natural language questions about the product — the LLM responds based on the retrieved records
