import altair as alt
import pandas as pd


df_names = pd.read_csv("data/interim/name_umap.csv", sep=",")
df_orgs = pd.read_csv("data/interim/org_umap.csv", sep=",")

chart_names = (

    alt.Chart(df_names)
    .mark_circle(size=30)
    .encode(
        x="x",
        y="y",
        tooltip=["nid", "name"],
    )
    .configure_mark(opacity=0.15, color="red")
    .configure_axis(grid=False)
    .properties(width=1200, height=1200)
    .interactive()
)

chart_names.save("vizs/name_embeddings.html")

chart_orgs = (
    alt.Chart(df_orgs)
    .mark_circle(size=30)
    .encode(
        x="x",
        y="y",
        tooltip=["nid", "name"],
    )
    .configure_mark(opacity=0.15, color="red")
    .configure_axis(grid=False)
    .properties(width=1200, height=1200)
    .interactive()
)

chart_orgs.save("vizs/org_embeddings.html")
