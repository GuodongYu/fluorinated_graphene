from fluorinated_G.fluorinated_graphene import Struct, diag_k_path
#from fluorinated_G.graphene import Struct, diag_k_path
from tBG.crystal.diagonalize import plot_band
st = Struct()
st.add_hopping()
st.add_Es_onsite()
diag_k_path(st)
plot_band(show=True)
