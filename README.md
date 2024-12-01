# Spatial Transcriptomics Analysis with Visium

Single-cell spatial transcriptomics is an emerging field, particularly in the study of tumor microenvironments in cancer. Recent advancements in technology, such as the Visium platform from 10x Genomics, enable the extraction of regional transcriptomic data alongside H&E staining from patient biopsy samples. In this example, we will use Scanpy to analyze single-cell RNA-seq data from the Visium platform, focusing on Glioblastoma.

First, we will import the necessary libraries and download the dataset:
```python
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

adata = sc.datasets.visium_sge(sample_id="Parent_Visium_Human_Glioblastoma")
adata.var_names_make_unique()
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
```

Next, we can plot the distribution of mitochondiral DNA to see which cells should be filtered out:
```python
sc.pl.violin(
    adata,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    jitter=0.4,
    multi_panel=True,
)
```
<img src="https://github.com/user-attachments/assets/e5dcb895-2fbd-4535-bec9-3111c1f4d2aa" alt="1" height = "200" width="600"/>

We can further filter on arbitrarily low or high gene expression:
```python
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
sns.histplot(adata.obs["total_counts"], kde=False, ax=axs[0])
sns.histplot(adata.obs["n_genes_by_counts"], kde=False, bins=60, ax=axs[1])

sc.pp.filter_cells(adata, min_counts=1500)
sc.pp.filter_cells(adata, max_counts=35000)
adata = adata[adata.obs["pct_counts_mt"] < 10].copy()
sc.pp.filter_genes(adata, min_cells=3)
print(f"#spots after filtering: {adata.n_obs}")
```
<img src="https://github.com/user-attachments/assets/78c21459-3bbf-42a9-8080-b183b7080fd9" alt="1" height = "200" width="400"/>

Then we normalize, identify the top genes leading to variation, and cluster:
```python
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)

sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(
    adata, key_added="clusters", flavor="igraph", directed=False, n_iterations=2
)

plt.rcParams["figure.figsize"] = (4, 4)
sc.pl.umap(adata, color=["total_counts", "n_genes_by_counts", "clusters"], wspace=0.4)
```
![image](https://github.com/user-attachments/assets/6a397764-721a-4f07-a277-61f432461c02)

Here we will visualize the spots from the Visium workflow:
```python
import numpy as np

hires_image = adata.uns["spatial"]["Parent_Visium_Human_Glioblastoma"]["images"]["hires"]
hires_image_uint8 = (hires_image * 255).astype(np.uint8)

spot_coords = adata.obsm["spatial"]
scaling_factor = adata.uns["spatial"]["Parent_Visium_Human_Glioblastoma"]["scalefactors"]["tissue_hires_scalef"]
scaled_coords = spot_coords * scaling_factor

fig, axs = plt.subplots(1, 2, figsize=(20, 10))

axs[0].imshow(hires_image_uint8)
axs[0].scatter(scaled_coords[:, 0], scaled_coords[:, 1], c="gray", s=5, label="Spots")
axs[0].axis("off")
axs[0].legend()

axs[1].imshow(hires_image_uint8)
axs[1].axis("off")

plt.show()
```
<img src="https://github.com/user-attachments/assets/be9e1d67-9dcc-495f-b431-63d666105b4d" alt="1" height = "300" width="600"/>

We can now color the spots based on the leiden clustering from above:
```python
sc.pl.spatial(adata, img_key="hires", color="clusters", size=1.5)
```
<img src="https://github.com/user-attachments/assets/756bc166-b923-4804-9e25-b43ce9b5330c" alt="1" height = "300" width="300"/>

Next we will visualize the gene expression across the clusters with a heatmap and a dotplot:
```python
sc.tl.rank_genes_groups(adata, "clusters", method="t-test")
sc.pl.rank_genes_groups_heatmap(adata, show_gene_labels=True)

sc.tl.rank_genes_groups(adata, groupby="clusters", method="wilcoxon")

sc.pl.rank_genes_groups_dotplot(
    adata, groupby="clusters", standard_scale="var", n_genes=5
)
```
![image](https://github.com/user-attachments/assets/9c7a46c9-056b-4d4d-9d2e-7c85a0acbe3b)
![image](https://github.com/user-attachments/assets/e69d9510-b47b-4964-a31d-ce6d6d8c6abc)

A closer inspection of the data revealed some patterns in clusters 4, 11, and 12:
```python
sc.tl.rank_genes_groups(adata, "clusters", method="t-test")
sc.pl.rank_genes_groups_heatmap(adata, groups=['4', '11', '12'], n_genes=15, groupby="clusters")
```
<img src="https://github.com/user-attachments/assets/9641a8a5-3e9a-422a-94ad-82a1064391b1" alt="1" height = "300" width="600"/>

Highlighting some top genes from these clusters:
```python
sc.pl.spatial(adata, img_key="hires", color=["clusters", 
                                             "VEGFA", 
                                             "ADM",
                                             'MT1X'], groups=["11", "4", "12"])
```
![image](https://github.com/user-attachments/assets/50a3db8e-f483-4fd2-b227-753b4a2282a2)

Next we rank the genes and then perform GSEA on the clusters:
```python
import pandas as pd
import gseapy as gp

sc.tl.rank_genes_groups(adata, groupby='clusters', method='wilcoxon')

ranked_genes = pd.DataFrame({
    'gene': adata.uns['rank_genes_groups']['names']['11'],
    'score': adata.uns['rank_genes_groups']['scores']['11']
})
ranked_genes.to_csv("cluster11_ranked_genes.csv", index=False)

ranked_genes = pd.DataFrame({
    'gene': adata.uns['rank_genes_groups']['names']['4'],
    'score': adata.uns['rank_genes_groups']['scores']['4']
})
ranked_genes.to_csv("cluster4_ranked_genes.csv", index=False)

ranked_genes = pd.DataFrame({
    'gene': adata.uns['rank_genes_groups']['names']['12'],
    'score': adata.uns['rank_genes_groups']['scores']['12']
})
ranked_genes.to_csv("cluster12_ranked_genes.csv", index=False)

with open("h.all.v2024.1.Hs.symbols.gmt", "r") as f:
    lines = f.readlines()

for line in lines[:5]:
    print(line.strip())

gsea_results = gp.prerank(
    rnk="cluster4_ranked_genes.csv",
    gene_sets="h.all.v2024.1.Hs.symbols.gmt",
    outdir="gsea_results_cluster4",  
    permutation_num=1000,
    min_size=15,
    max_size=500,
)

gsea_results = gp.prerank(
    rnk="cluster11_ranked_genes.csv",  
    gene_sets="h.all.v2024.1.Hs.symbols.gmt",
    outdir="gsea_results_cluster11", 
    permutation_num=1000, 
    min_size=15,
    max_size=500,
)

gsea_results = gp.prerank(
    rnk="cluster12_ranked_genes.csv", 
    gene_sets="h.all.v2024.1.Hs.symbols.gmt",
    outdir="gsea_results_cluster12",
    permutation_num=1000,
    min_size=15,
    max_size=500,
)
```
After inspecting the results, the Hypoxia Hallmark Pathway is upregulated in pretty much all three clusters. As a result, we can create a custom entry in our AnnData object for the gene list by downloading geneset .json from GSEA:
```python
import json

with open('HALLMARK_HYPOXIA.v2024.1.Hs.json') as f:
    data = json.load(f)

geneset_hypoxia = data["HALLMARK_HYPOXIA"]["geneSymbols"]

geneset_in_data = [gene for gene in geneset_hypoxia if gene in adata.var_names]

sc.tl.score_genes(adata, gene_list=geneset_in_data, score_name='hypoxia_score')

sc.pl.spatial(adata, img_key="hires", color=["clusters","hypoxia_score"], groups=["11", "4", "12"])
```
<img src="https://github.com/user-attachments/assets/d45dd221-35be-4c2d-9409-7bba7b7f6284" alt="1" height = "300" width="600"/>

As you can see, we were able to spatially identify regions of the tumor microenviornment associated with hypoxia. We could do some more interesting analysis, but I will leave it to you!
