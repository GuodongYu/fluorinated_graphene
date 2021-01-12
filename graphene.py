import numpy as np
from tBTMDC.tmdc_twisted_bilayer import get_local_axes, get_direct_cosine_with_local_axes
from tBG.utils import rotate_on_vec
from tBG.crystal.brillouin_zones import BZHexagonal
from scipy.linalg.lapack import zheev

class Struct:
    def __init__(self, a=2.47):
        """
        structure descriptiong:
        armchair and zigzag are along y- and x-axis, respectively.
        atom order: C0, C1, F0, F1, where F0 and F1 are attached to C0 and C1, respectively.
                    F0 and C0 bend towards z axis, F1 and V1 bend towards -z axis.

        vector delta_1 is C0(0,0) -> C1( 0,  0)  (0,0) stands for the unit cell
        vector delta_2 is C0(0,0) -> C1(-1,  0)
        vector delta_3 is C0(0,0) -> C1( 0, -1)
        vector delta_F is C0->F0 or C1->F1 
        
        orbitals include C: s, px, py, pz
                         F: px, py, pz   
        """
        self.a = a
        self.latt_vec = a*np.array([[1/2, np.sqrt(3)/2, 0],[-1/2, np.sqrt(3)/2, 0], [0, 0, 100/a]])
        delta = np.array([0, self.a/np.sqrt(3)])
        self.deltas = np.array([rotate_on_vec(120*i, delta) for i in range(3)])
        self.lmn = {'delta_1':[0, 1, 0], 'delta_2':[-np.sqrt(3)/2, -1/2, 0],\
                    'delta_3':[np.sqrt(3)/2, -1/2, 0]}
        self.hopping_params = {'C-C':{'V_sssigma':-5.34, 'V_spsigma':6.4, \
                                      'V_pxypxypi':-2.8,'V_pxypxysigma':7.65,\
                                      'V_pzpzpi':-2.8, 'V_pzpzsigma':0,\
                                      'V_pxypzpi':0, 'V_pxypzsigma':0}}
        self.E_onsite_params = {'C':{'s':-2.85, 'pxy':3.2, 'pz':0.0}} 

    def _calc_pair_hop(self, V_sigma, V_pi, vec_orb0, vec_orb1, lmn_1to0):
        local_axes = get_local_axes(lmn_1to0)
        lmn0_local = get_direct_cosine_with_local_axes(vec_orb0, local_axes)
        lmn1_local = get_direct_cosine_with_local_axes(vec_orb1, local_axes)
        return np.sum(np.array(lmn0_local)*np.array(lmn1_local)*np.array([V_sigma, V_pi, V_pi]))


    def add_hopping(self):
        hoppings = [ {} for i in range(4)]

        ### C0->C1 ###
        ### from  C0 s, px, py, pz
        ###  to   C1 s, px, py, pz ###
        lmns = [self.lmn['delta_1'], self.lmn['delta_2'], self.lmn['delta_3']]
        ucs = [[0,0], [-1,0], [0, -1]] 
        hop = self.hopping_params['C-C']
        # loop for three different hopping [00] to [00], [-1,0] and [0,-1] unit cells
        for i in range(3):
            lmn = np.array(lmns[i])
            t = np.zeros([4,4])
            uc = ucs[i]
            t[0,0] = hop['V_sssigma']
            t[0,1] = self._calc_pair_hop(hop['V_spsigma'], 0, lmn, [1, 0, 0], lmn)
            t[0,2] = self._calc_pair_hop(hop['V_spsigma'], 0, lmn, [0, 1, 0], lmn)
            t[0,3] = self._calc_pair_hop(hop['V_spsigma'], 0, lmn, [0, 0, 1], lmn)
            t[1,0] = self._calc_pair_hop(hop['V_spsigma'], 0, [1,0,0], -lmn, lmn)
            t[1,1] = self._calc_pair_hop(hop['V_pxypxysigma'], hop['V_pxypxypi'], [1,0,0], [1,0,0], lmn)
            t[1,2] = self._calc_pair_hop(hop['V_pxypxysigma'], hop['V_pxypxypi'], [1,0,0], [0,1,0], lmn)
            t[1,3] = self._calc_pair_hop(hop['V_pxypzsigma'], hop['V_pxypzpi'], [1,0,0], [0,0,1], lmn)
            t[2,0] = self._calc_pair_hop(hop['V_spsigma'], 0, [0,1,0], -lmn, lmn)
            t[2,1] = self._calc_pair_hop(hop['V_pxypxysigma'], hop['V_pxypxypi'], [0,1,0], [1,0,0], lmn)
            t[2,2] = self._calc_pair_hop(hop['V_pxypxysigma'], hop['V_pxypxypi'], [0,1,0], [0,1,0], lmn)
            t[2,3] = self._calc_pair_hop(hop['V_pxypzsigma'], hop['V_pxypzpi'], [0,1,0], [0,0,1], lmn)
            t[3,0] = self._calc_pair_hop(hop['V_spsigma'], 0, [0,0,1], -lmn, lmn)
            t[3,1] = self._calc_pair_hop(hop['V_pxypzsigma'], hop['V_pxypzpi'], [0,0,1], [1,0,0], lmn)
            t[3,2] = self._calc_pair_hop(hop['V_pxypzsigma'], hop['V_pxypzpi'], [0,0,1], [0,1,0], lmn)
            t[3,3] = self._calc_pair_hop(hop['V_pzpzsigma'], hop['V_pzpzpi'], [0,0,1], [0,0,1], lmn)
            hoppings[0][(uc[0],uc[1],1)] = np.array(t)
        self.hopping = hoppings

    def add_Es_onsite(self):
        e_C = self.E_onsite_params['C'] 
        Es_onsite = [e_C['s'], e_C['pxy'], e_C['pxy'], e_C['pz']]*2  
        self.Es_onsite = np.array(Es_onsite)

    def get_Hamiltonian_at_k(self, k):
        k = np.array(k)
        Hk = np.zeros([8, 8], dtype=complex)
        np.fill_diagonal(Hk, self.Es_onsite)

        ### C0 (0,0) ---> C1 (00)
        ucs = [(0,0,1),(-1,0,1),(0,-1,1)]
        for i in range(3):
            delta = self.deltas[i]
            uc = ucs[i]
            tk = np.exp(1j*np.dot(k, delta[0:2]))* self.hopping[0][uc]
            Hk[0:4, 4:8] = Hk[0:4, 4:8] + tk
            Hk[4:8, 0:4] = Hk[4:8, 0:4] + np.transpose(tk).conj()
        return Hk

    def diag_kpts(self, kpts, vec=0):
        """
        kpts: the coordinates of kpoints
        vec: whether to calculate the eigen vectors
        fname: the file saveing results
        """
        val_out = []
        vec_out = []
        i = 1
        if vec:
            vec_calc = 1
        else:
            vec_calc = 0
        for k in kpts:
            print('%s/%s k' % (i, len(kpts)))
            Hk = self.get_Hamiltonian_at_k(k)
            vals, vecs, info = zheev(Hk, vec_calc)
            if info:
                raise ValueError('zheev failed')
            val_out.append(vals)
            if vec:
                vec_out.append(vecs)
            i = i + 1
        return np.array(val_out), np.array(vec_out)
            
def diag_k_path(struct, k_path=['G','K','M','G'], dk=0.01, fname='EIGEN_val_path'):
    """
    This function is used to calculate band structure along k_path
    """
    bz = BZHexagonal(struct.latt_vec)
    kpts, inds = bz.kpoints_line_mode(k_path, dk)
    vals, vecs = struct.diag_kpts(kpts, vec=0)
    k_info = {'labels':k_path, 'inds':inds}
    np.savez_compressed(fname, kpoints=kpts, vals=vals, k_info=[k_info])


def plot_band(st):
    pass

