import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    return mo, np


@app.cell
def _(mo):
    random_state_slider = mo.ui.slider(0,100)
    w1_slide = mo.ui.slider(-2,2,0.1)
    w2_slide = mo.ui.slider(-2,2,0.1)
    b_slide = mo.ui.slider(-10,10,0.5)
    return b_slide, random_state_slider, w1_slide, w2_slide


@app.cell
def _(mo, random_state_slider):
    mo.md(f"""
    # Illustration d'une procédure d'optimisation de paramètres
    seed pour fixer la génération aléatoire des données : {random_state_slider} {random_state_slider.value}
    """)
    return


@app.cell
def _(b_slide, mo, w1_slide, w2_slide):
    mo.md(f"""
    ## Paramètres à optimiser : 

    $W_1$ = {w1_slide.value} {w1_slide}

    $W_2$ = {w2_slide.value} {w2_slide}

    $b$ = {b_slide.value} {b_slide}
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Fonction "apprise"
    $$ f(x) = x^2 \times W_2 + x \times W_1 + b $$
    """)
    return


@app.cell
def _(np, random_state_slider):
    np.random.seed(random_state_slider.value)
    coefs = np.random.normal(size=3)
    def f(x):
        return sum([coefs[i]*(x**i) for i in range(len(coefs))])

    X = np.random.sample(size=50)*6 -3
    Y = np.array([f(x) for x in X])
    noise = np.random.normal(0,0.5,size=len(X))
    Y = Y + noise
    return X, Y


@app.cell
def _():
    #coefs
    return


@app.cell
def _(X, Y, b_slide, mo, np, w1_slide, w2_slide):
    import altair as alt
    import polars as pl

    df = pl.DataFrame(data={'x': X, 'y': Y})

    chart = (
        alt.Chart(df)
        .mark_point()
        .encode(
            x="x",
            y="y",
        )

    )
    #chart = chart + chart.mark_line(color="green").encode(x="x", y="yth")

    def learned_f(x):
        return x*w1_slide.value + (x**2)*w2_slide.value + b_slide.value

    X_poly = np.linspace(X.min() , X.max(),20)
    df_poly = pl.DataFrame({'x':X_poly, 'y':learned_f(X_poly) })
    line = (
        alt.Chart(df_poly)
        .mark_line(color="red")
        .encode(x="x", y="y")
    )


    chart = mo.ui.altair_chart(chart + line)
    return alt, chart, learned_f, pl


@app.cell
def _(mo):
    mo.md(r"""
    ## Donnée observées (bleu) et fonction apprise (rouge)
    """)
    return


@app.cell
def _(chart):
    chart
    return


@app.cell
def _(X, Y, learned_f, np):
    MSE = np.mean([(Y[i] - learned_f(X[i]))**2 for i in range(len(X))])
    return (MSE,)


@app.cell
def _(MSE, mo):
    mo.md(f"""
    # Calcul de la Loss

    $$ Loss: MSE = {MSE:0.4f}$$

    ## Décomposition de la Loss

    Calcul de la loss en fonction de la valeur d'un des paramètres 

    (en fixant les deux autres)
    """)
    return


@app.cell
def _(X, Y, alt, b_slide, np, pl, w1_slide, w2_slide):
    def loss_w1(v):
        def fw1(x,v):
            return x*v + (x**2)*w2_slide.value + b_slide.value
        return  np.mean([(Y[i] - fw1(X[i],v))**2 for i in range(len(X))])

    V = np.linspace(-2,2,30)
    Yw1 = [loss_w1(x) for x in V]


    _df = pl.DataFrame(data={'w1': V, 'Loss': Yw1})

    _chart = (
        alt.Chart(_df)
        .mark_line()
        .encode(
            x="w1",
            y="Loss",
        )

    )
    _chart + alt.Chart(pl.DataFrame(data={'v':[w1_slide.value]})).mark_rule().encode(x='v')
    return


@app.cell
def _(X, Y, alt, b_slide, np, pl, w1_slide, w2_slide):
    def loss_w2(v):
        def fw2(x,v):
            return (x**2)*v + x*w1_slide.value + b_slide.value
        return  np.mean([(Y[i] - fw2(X[i],v))**2 for i in range(len(X))])

    _V = np.linspace(-2,2,30)
    _Yw2 = [loss_w2(x) for x in _V]


    _df = pl.DataFrame(data={'w2': _V, 'Loss': _Yw2})

    _chart = (
        alt.Chart(_df)
        .mark_line()
        .encode(
            x="w2",
            y="Loss",
        )

    )
    _chart + alt.Chart(pl.DataFrame(data={'v':[w2_slide.value]})).mark_rule().encode(x='v')
    return


@app.cell
def _(X, Y, alt, b_slide, np, pl, w1_slide, w2_slide):
    def loss_b(v):
        def fb(x,v):
            return (x**2)*w2_slide.value + x*w1_slide.value + v
        return  np.mean([(Y[i] - fb(X[i],v))**2 for i in range(len(X))])

    _V = np.linspace(-10,10,50)
    _Yb = [loss_b(x) for x in _V]


    _df = pl.DataFrame(data={'b': _V, 'Loss': _Yb})

    _chart = (
        alt.Chart(_df)
        .mark_line()
        .encode(
            x="b",
            y="Loss",
        )

    )
    _chart + alt.Chart(pl.DataFrame(data={'v':[b_slide.value]})).mark_rule().encode(x='v')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
