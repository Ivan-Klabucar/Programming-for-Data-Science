# %%
from rdkit import Chem
import rdkit.Chem.rdMolDescriptors as d
import rdkit.Chem.Fragments as f
import rdkit.Chem.Lipinski as lp
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from rdkit.Chem import AllChem

# %%
df = pd.read_csv("training_smiles.csv")
df['mol'] = df['SMILES'].apply(lambda x: Chem.MolFromSmiles(x))
# %% features
featuresFn = [
    ('FractionCSP3', (lambda x: lp.FractionCSP3(x))),
    ('HeavyAtomCount', (lambda x: lp.HeavyAtomCount(x))),
    ('NHOHCount', (lambda x: lp.NHOHCount(x))),
    ('NOCount', (lambda x: lp.NOCount(x))),
    ('NumAliphaticCarbocycles', (lambda x: lp.NumAliphaticCarbocycles(x))),
    ('NumAliphaticHeterocycles', (lambda x: lp.NumAliphaticHeterocycles(x))),
    ('NumAliphaticRings', (lambda x: lp.NumAliphaticRings(x))),
    ('NumAromaticCarbocycles', (lambda x: lp.NumAromaticCarbocycles(x))),
    ('NumAromaticHeterocycles', (lambda x: lp.NumAromaticHeterocycles(x))),
    ('NumAromaticRings', (lambda x: lp.NumAromaticRings(x))),
    ('NumHAcceptors', (lambda x: lp.NumHAcceptors(x))),
    ('NumHDonors', (lambda x: lp.NumHDonors(x))),
    ('NumHeteroatoms', (lambda x: lp.NumHeteroatoms(x))),
    ('NumRotatableBonds', (lambda x: lp.NumRotatableBonds(x))),
    ('NumSaturatedCarbocycles', (lambda x: lp.NumSaturatedCarbocycles(x))),
    ('NumSaturatedHeterocycles', (lambda x: lp.NumSaturatedHeterocycles(x))),
    ('NumSaturatedRings', (lambda x: lp.NumSaturatedRings(x))),
    ('RingCount', (lambda x: lp.RingCount(x))),

    ('fr_Al_COO', (lambda x: f.fr_Al_COO(x) )),
    ('fr_Al_OH', (lambda x: f.fr_Al_OH(x) )),
    ('fr_Al_OH_noTert', (lambda x: f.fr_Al_OH_noTert(x) )),
    ('fr_ArN', (lambda x: f.fr_ArN(x) )),
    ('fr_Ar_COO', (lambda x: f.fr_Ar_COO(x) )),
    ('fr_Ar_N', (lambda x: f.fr_Ar_N(x) )),
    ('fr_Ar_NH', (lambda x: f.fr_Ar_NH(x) )),
    ('fr_Ar_OH', (lambda x: f.fr_Ar_OH(x) )),
    ('fr_COO', (lambda x: f.fr_COO(x) )),
    ('fr_COO2', (lambda x: f.fr_COO2(x) )),
    ('fr_C_O', (lambda x: f.fr_C_O(x) )),
    ('fr_C_O_noCOO', (lambda x: f.fr_C_O_noCOO(x) )),
    ('fr_C_S', (lambda x: f.fr_C_S(x) )),
    ('fr_HOCCN', (lambda x: f.fr_HOCCN(x) )),
    ('fr_Imine', (lambda x: f.fr_Imine(x) )),
    ('fr_NH0', (lambda x: f.fr_NH0(x) )),
    ('fr_NH1', (lambda x: f.fr_NH1(x) )),
    ('fr_NH2', (lambda x: f.fr_NH2(x) )),
    ('fr_N_O', (lambda x: f.fr_N_O(x) )),
    ('fr_Ndealkylation1', (lambda x: f.fr_Ndealkylation1(x) )),
    ('fr_Ndealkylation2', (lambda x: f.fr_Ndealkylation2(x) )),
    ('fr_Nhpyrrole', (lambda x: f.fr_Nhpyrrole(x) )),
    ('fr_SH', (lambda x: f.fr_SH(x) )),
    ('fr_aldehyde', (lambda x: f.fr_aldehyde(x) )),
    ('fr_alkyl_carbamate', (lambda x: f.fr_alkyl_carbamate(x) )),
    ('fr_alkyl_halide', (lambda x: f.fr_alkyl_halide(x) )),
    ('fr_allylic_oxid', (lambda x: f.fr_allylic_oxid(x) )),
    ('fr_amide', (lambda x: f.fr_amide(x) )),
    ('fr_amidine', (lambda x: f.fr_amidine(x) )),
    ('fr_aniline', (lambda x: f.fr_aniline(x) )),
    ('fr_aryl_methyl', (lambda x: f.fr_aryl_methyl(x) )),
    ('fr_azide', (lambda x: f.fr_azide(x) )),
    ('fr_azo', (lambda x: f.fr_azo(x) )),
    ('fr_barbitur', (lambda x: f.fr_barbitur(x) )),
    ('fr_benzene', (lambda x: f.fr_benzene(x) )),
    ('fr_benzodiazepine', (lambda x: f.fr_benzodiazepine(x) )),
    ('fr_bicyclic', (lambda x: f.fr_bicyclic(x) )),
    ('fr_diazo', (lambda x: f.fr_diazo(x) )),
    ('fr_dihydropyridine', (lambda x: f.fr_dihydropyridine(x) )),
    ('fr_epoxide', (lambda x: f.fr_epoxide(x) )),
    ('fr_ester', (lambda x: f.fr_ester(x) )),
    ('fr_ether', (lambda x: f.fr_ether(x) )),
    ('fr_furan', (lambda x: f.fr_furan(x) )),
    ('fr_guanido', (lambda x: f.fr_guanido(x) )),
    ('fr_halogen', (lambda x: f.fr_halogen(x) )),
    ('fr_hdrzine', (lambda x: f.fr_hdrzine(x) )),
    ('fr_hdrzone', (lambda x: f.fr_hdrzone(x) )),
    ('fr_imidazole', (lambda x: f.fr_imidazole(x) )),
    ('fr_imide', (lambda x: f.fr_imide(x) )),
    ('fr_isocyan', (lambda x: f.fr_isocyan(x) )),
    ('fr_isothiocyan', (lambda x: f.fr_isothiocyan(x) )),
    ('fr_ketone', (lambda x: f.fr_ketone(x) )),
    ('fr_ketone_Topliss', (lambda x: f.fr_ketone_Topliss(x) )),
    ('fr_lactam', (lambda x: f.fr_lactam(x) )),
    ('fr_lactone', (lambda x: f.fr_lactone(x) )),
    ('fr_methoxy', (lambda x: f.fr_methoxy(x) )),
    ('fr_morpholine', (lambda x: f.fr_morpholine(x) )),
    ('fr_nitrile', (lambda x: f.fr_nitrile(x) )),
    ('fr_nitro', (lambda x: f.fr_nitro(x) )),
    ('fr_nitro_arom', (lambda x: f.fr_nitro_arom(x) )),
    ('fr_nitro_arom_nonortho', (lambda x: f.fr_nitro_arom_nonortho(x) )),
    ('fr_nitroso', (lambda x: f.fr_nitroso(x) )),
    ('fr_oxazole', (lambda x: f.fr_oxazole(x) )),
    ('fr_oxime', (lambda x: f.fr_oxime(x) )),
    ('fr_para_hydroxylation', (lambda x: f.fr_para_hydroxylation(x) )),
    ('fr_phenol', (lambda x: f.fr_phenol(x) )),
    ('fr_phenol_noOrthoHbond', (lambda x: f.fr_phenol_noOrthoHbond(x) )),
    ('fr_phos_acid', (lambda x: f.fr_phos_acid(x) )),
    ('fr_phos_ester', (lambda x: f.fr_phos_ester(x) )),
    ('fr_piperdine', (lambda x: f.fr_piperdine(x) )),
    ('fr_piperzine', (lambda x: f.fr_piperzine(x) )),
    ('fr_priamide', (lambda x: f.fr_priamide(x) )),
    ('fr_prisulfonamd', (lambda x: f.fr_prisulfonamd(x) )),
    ('fr_pyridine', (lambda x: f.fr_pyridine(x) )),
    ('fr_quatN', (lambda x: f.fr_quatN(x) )),
    ('fr_sulfide', (lambda x: f.fr_sulfide(x) )),
    ('fr_sulfonamd', (lambda x: f.fr_sulfonamd(x) )),
    ('fr_sulfone', (lambda x: f.fr_sulfone(x) )),
    ('fr_term_acetylene', (lambda x: f.fr_term_acetylene(x) )),
    ('fr_tetrazole', (lambda x: f.fr_tetrazole(x) )),
    ('fr_thiazole', (lambda x: f.fr_thiazole(x) )),
    ('fr_thiocyan', (lambda x: f.fr_thiocyan(x) )),
    ('fr_thiophene', (lambda x: f.fr_thiophene(x) )),
    ('fr_unbrch_alkane', (lambda x: f.fr_unbrch_alkane(x) )),
    ('fr_urea', (lambda x: f.fr_urea(x) ))
]

# %% 
df['mol'].apply(lambda row: fn(row['mol']), axis='columns', result_type='expand')

morgan = df['mol'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2,nBits=124), axis='columns', result_type='expand')
print(morgan.shape)

# %%
for colName, fn in tqdm(featuresFn):
    try:
        df[colName] = df['mol'].apply(fn)
    except:
        print("Error in feature: ", colName)
df = df.copy()

# %%
df.corr().style.background_gradient(cmap='coolwarm')

# %%

pca = PCA(0.95)
dfFeatures = df.drop(['mol', 'INDEX', 'SMILES', 'ACTIVE'], axis=1)
pca.fit(dfFeatures)
pcaDf = pd.DataFrame(pca.transform(dfFeatures))
pcaDf['labels'] = df['ACTIVE']
print(pcaDf.shape)


# %%
