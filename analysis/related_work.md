# Related work (Step 8 literature review, in progress)

**Verification status.** "Verified" means the bibliographic details below were confirmed by a web search in this session (2026-10-10), with the link given. "Repo bib" means the entry is taken from `paper/references.bib`. "Standard" means a widely cited classic whose details are from standard knowledge and **should be checked against the publisher before submission**.

## 1. What prior work says about our hypotheses

| Our hypothesis | Closest prior evidence | Implication for this study |
|---|---|---|
| H1/H3: does news **sentiment (tone)** add to HAR? | **Mixed.** Bitcoin: news sentiment improves HAR-RV fit and forecasts (Sapkota 2022), but a 2024 crypto study finds that AI-generated news sentiment does *not* improve HAR, only non-linear ML (Springer 2024). Oil: FinBERT, VADER and others performed *worse* than HAR (arXiv 2508.20707). US equities: FinBERT macro-news sentiment helps in a complete-subset HAR framework (arXiv 2503.19767). | A null result for tone in a linear HAR is plausible and publishable. The pre-registered MDE reporting (PREREGISTRATION §5) is essential to interpret a null. |
| H5 (ATTN): does news **volume/attention** help? | Attention (Google searches, StockTwits message volume) was the strongest predictor of realized volatility after extensive controls (Audrino, Sigrist & Ballinari 2020). The raw news count was robust where tone measures failed (arXiv 2508.20707). | Supports keeping ATTN as a separate pre-registered group. Attention may matter more than tone. |
| H2/H3 (non-linear models) | ML (regularized, trees, neural nets) is competitive with and can beat HAR even with only HAR lags (Christensen, Siggaard & Veliyev 2023). LOB data help more than news sentiment in LSTMs; news adds marginally (Rahimikia & Poon). | The HAR-only GBM/Neural variants (GBM-HAR, NN-GRN-HAR) are necessary to separate "non-linearity" from "sentiment". Both are pre-registered. |
| Conditional effects (Step 8, exploratory) | News embeddings help mainly on **volatility-jump days** (Rahimikia, Zohren & Poon 2021/2026). | Motivates the exploratory high-volatility / high-news conditional analysis. It is exploratory and cannot enter the primary decision rule (PREREGISTRATION §9). |
| H4 (calibration, QLIKE) | QLIKE is robust to noise in the volatility proxy (Patton 2011). HARQ lets coefficients vary with measurement error (Bollerslev, Patton & Quaedvlieg 2016). | Justifies QLIKE as primary and HARQ as a BTC baseline. Our Phase-1 finding (smearing closes most of the QLIKE gap) is a calibration effect that much of the ML-vs-HAR literature does not control for. Worth stating explicitly in the paper. |

## 2. References

### Volatility modelling and forecast evaluation
* Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174–196. *(repo bib)*
* Bollerslev, T., Patton, A. J., & Quaedvlieg, R. (2016). Exploiting the errors: A simple approach for improved volatility forecasting. *Journal of Econometrics*, 192(1), 1–18. doi:10.1016/j.jeconom.2015.10.007. *(verified: [Maastricht CRIS](https://cris.maastrichtuniversity.nl/en/publications/exploiting-the-errors-a-simple-approach-for-improved-volatility-f/))*
* Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*, 160(1), 246–256. *(verified: [Duke Scholars](https://scholars.duke.edu/publication/1072550); repo bib)*
* Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263. *(repo bib)*
* Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281–291. *(repo bib)*
* Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set. *Econometrica*, 79(2), 453–497. *(repo bib)*
* Newey, W. K., & West, K. D. (1994). Automatic lag selection in covariance matrix estimation. *Review of Economic Studies*, 61(4), 631–653. *(standard)*
* Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303–1313. *(standard)*
* Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70. *(standard)*
* Garman, M. B., & Klass, M. J. (1980). On the estimation of security price volatilities from historical data. *Journal of Business*, 53(1), 67–78. *(standard)*
* Duan, N. (1983). Smearing estimate: A nonparametric retransformation method. *Journal of the American Statistical Association*, 78(383), 605–610. *(standard)*
* Barndorff-Nielsen, O. E., & Shephard, N. (2002). Econometric analysis of realized volatility and its use in estimating stochastic volatility models. *Journal of the Royal Statistical Society B*, 64(2), 253–280. *(standard)*

### Machine learning for realized volatility
* Christensen, K., Siggaard, M., & Veliyev, B. (2023). A machine learning approach to volatility forecasting. *Journal of Financial Econometrics*, 21(5), 1680–1727. doi:10.1093/jjfinec/nbac020. *(verified: [CREATES RP 2021-03](https://repec.econ.au.dk/repec/creates/rp/21/rp21_03.pdf), [x-mol](https://ma.x-mol.com/paper/1539803485281558528))*
* Rahimikia, E., Zohren, S., & Poon, S.-H. (2021; rev. 2026). Realised volatility forecasting: Machine learning via financial word embedding. arXiv:2108.00480. *(verified: [arXiv](https://arxiv.org/abs/2108.00480); EFMA 2022 version [pdf](https://www.efmaefm.org/0EFMAMEETINGS/EFMA%20ANNUAL%20MEETINGS/2022-Rome/papers/EFMA%202022_stage-3032_question-Full%20Paper_id-9.pdf); journal venue not verified)*
* Rahimikia, E., & Poon, S.-H. Big data approach to realised volatility forecasting using HAR model augmented with limit order book and news. *(described in the sources above; exact venue/year not verified, check before citing)*
* Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). Revisiting deep learning models for tabular data. *NeurIPS 2021*. *(repo bib)*
* Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting*, 37(4), 1748–1764. *(standard; source of the GRN, missing from the repo bib, see paper_vs_code.md C9)*

### Sentiment, attention and volatility
* Audrino, F., Sigrist, F., & Ballinari, D. (2020). The impact of sentiment and attention measures on stock market volatility. *International Journal of Forecasting*, 36(2), 334–357. *(verified: [RePEc/IDEAS](https://ideas.repec.org/f/pba1855.html), [HSG Alexandria](https://alexandria.unisg.ch/handle/20.500.14171/112549))*
* Sapkota, N. (2022). News-based sentiment and bitcoin volatility. *International Review of Financial Analysis*. *(verified via [Univ. of Vaasa repository](https://osuva.uwasa.fi/handle/10024/15172); volume/pages not retrieved)*
* Crypto-market news sentiment, HAR vs ML (2024). *Asia-Pacific Financial Markets* (Springer), doi:10.1007/s10690-024-09510-6. *(verified link [Springer](https://link.springer.com/article/10.1007/s10690-024-09510-6); authors not retrieved)*
* Forecasting U.S. equity market volatility with attention and sentiment to the economy (2025). arXiv:2503.19767. *(verified link [arXiv](https://arxiv.org/html/2503.19767v1); venue/authors not retrieved)*
* Can news predict the direction of oil price volatility? (2025). arXiv:2508.20707. *(verified link [arXiv](https://www.arxiv.org/pdf/2508.20707))*
* Tetlock, P. C. (2007). Giving content to investor sentiment: The role of media in the stock market. *Journal of Finance*, 62(3), 1139–1168. *(standard)*
* Loughran, T., & McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. *Journal of Finance*, 66(1), 35–65. *(standard)*

### Text models and data
* Araci, D. (2019). FinBERT: Financial sentiment analysis with pre-trained language models. arXiv:1908.10063. *(repo bib)*
* Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. *Proceedings of ICWSM*. *(standard)*
* Leetaru, K., & Schrodt, P. A. (2013). GDELT: Global data on events, location, and tone, 1979–2012. *ISA Annual Convention*. *(standard)*

### Microstructure features
* Easley, D., López de Prado, M. M., & O'Hara, M. (2012). Flow toxicity and liquidity in a high-frequency world. *Review of Financial Studies*, 25(5), 1457–1493. *(standard)*
* Amihud, Y. (2002). Illiquidity and stock returns: Cross-section and time-series effects. *Journal of Financial Markets*, 5(1), 31–56. *(standard)*
* Roll, R. (1984). A simple implicit measure of the effective bid-ask spread in an efficient market. *Journal of Finance*, 39(4), 1127–1139. *(standard)*

## 3. Still to do (Step 8)
* Systematic search (Google Scholar/SSRN are not reachable from the sandbox; web search only) for 2023–2026 work on GDELT-based tone and volatility, and on India/NIFTY news sentiment.
* A BibTeX block for every reference, generated once the reference list is final.
