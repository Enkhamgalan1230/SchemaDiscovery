import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Table Relation App",
    page_icon="🔗",
    layout="wide"
)

root = Path(__file__).parent

# ---------- NAVIGATION ----------
data_schematics = st.Page("page_numbers/table_relation.py", title="Data Schematics",icon=":material/dataset:")
profiler = st.Page("page_numbers/extras.py", title="Data Schematics Components",icon=":material/star:")
enhanced_profiler = st.Page("page_numbers/enhanced_profiler.py", title="Enhanced Profiler",icon=":material/star_shine:")
missing = st.Page("page_numbers/missing.py", title="Missingness detection",icon=":material/dangerous:")
duplicates = st.Page("page_numbers/dup.py", title="Duplicates detection",icon=":material/content_copy:")
default = st.Page("page_numbers/default.py", title="Default FK detection",icon=":material/award_star:")
surrogate = st.Page("page_numbers/surrogate.py", title="Surrogate Key detection",icon=":material/format_list_numbered_rtl:")
composite = st.Page("page_numbers/composite.py", title="Composite Key detection",icon=":material/merge:")
recom = st.Page("page_numbers/recommendation.py", title="Recommendation Engine",icon=":material/reviews:")

pg = st.navigation({
    "Main Pipeline": [data_schematics],
    "Extra": [profiler,enhanced_profiler, missing, duplicates,default,surrogate,composite],
    "Recommendation": [recom]

})

pg.run()

