import streamlit as st

from params import get_fragment_spectra_params
from utils import get_fragment_spectra_plot, get_fragment_ions_df, convert_df

st.header("Fragment Ion Calculator")

seq_col, charge_col = st.columns(2)
with seq_col:
    peptide = st.text_input('Peptide Sequence', 'PEPTIDE')
with charge_col:
    charge = st.number_input('Charge:', min_value=1, max_value=10, value=1,
                             help="peptide charge state to use for fragmentation")

fragment_spectra_params = get_fragment_spectra_params()
fig = get_fragment_spectra_plot(peptide, charge, fragment_spectra_params)
st.plotly_chart(fig)

frag_df = get_fragment_ions_df(peptide, charge, fragment_spectra_params)
with st.expander("All Ions"):
    st.table(frag_df)
for charge, grp in frag_df.groupby(by=['charge']):
    charge_df = grp.reset_index()
    if charge == 0:
        st.subheader(f"Immonium Ions")
        charge_df = charge_df[['ion', 'mz']]
        charge_df['ion'] = [ion[-1] for ion in charge_df.ion]
        st.table(charge_df)
        continue
    charge_df = charge_df[charge_df.ion_type.isin(['a', 'b', 'c', 'x', 'y', 'z'])]
    st.subheader(f"Charge {charge}")
    piv_df = charge_df.pivot_table('mz', index='ion_num', columns='ion_tag')
    cols = piv_df.columns
    forward_cols = sorted(list(set(cols) - {'x', 'y', 'z', 'x-H20', 'y-H20', 'z-H20', 'x-H3N', 'y-H3N', 'z-H3N'}))
    backward_cols = sorted(list(set(cols) - {'a', 'b', 'c', 'a-H20', 'b-H20', 'c-H20', 'a-H3N', 'b-H3N', 'c-H3N'}))

    forward_ion_df = piv_df[forward_cols]
    backward_ion_df = piv_df[backward_cols]

    col1, col2 = st.columns(2)
    col1.table(forward_ion_df)
    col2.table(backward_ion_df.reindex(index=backward_ion_df.index[::-1]))

st.download_button(label="Download Frag Ions",
                   data=convert_df(frag_df),
                   file_name='frag_ions.csv',
                   mime='text/csv', )