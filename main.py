import streamlit as st

from params import get_fragment_spectra_params
from utils import get_fragment_spectra_plot, get_fragment_ions_df, convert_df

st.header("Fragment Ion Calculator")

seq_col, charge_col = st.columns(2)
with seq_col:
    peptide = st.text_input('Peptide Sequence', 'PEPTIDE')
with charge_col:
    charge = st.number_input('Charge:', min_value=0, max_value=1000, value=1,
                             help="peptide charge state to use for fragmentation")

fragment_spectra_params = get_fragment_spectra_params()
fig = get_fragment_spectra_plot(peptide, charge, fragment_spectra_params)
st.plotly_chart(fig)

frag_df = get_fragment_ions_df(peptide, charge, fragment_spectra_params)[['ion', 'mz', 'intensity']]
st.dataframe(frag_df)

st.download_button(label="Download Frag Ions",
                   data=convert_df(frag_df),
                   file_name='frag_ions.csv',
                   mime='text/csv', )