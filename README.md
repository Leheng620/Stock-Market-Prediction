# Stock-Market-Prediction

# Models
Some of the SOTA models and techniques used for predicting stock trends and prices include:

1. Prophet: A time series forecasting model developed by Facebook that uses additive models to forecast time series data based on an underlying trend and seasonal components, as well as user-provided input features and special events.
2. XGBoost: An implementation of gradient boosting decision trees that can be used for regression tasks, such as predicting stock prices. XGBoost is known for its high performance and scalability.
3. Random Forest: Another decision tree-based ensemble model that can be used for regression tasks, such as predicting stock prices. Random forests can handle noisy data and are good at capturing nonlinear relationships between the input features and output variable.


# Datasets:
1. [2008-2016的新闻数据](https://www.kaggle.com/datasets/aaron7sun/stocknews)
  

# Methods
大家可以看看[这个](https://paperswithcode.com/search?q_meta=&q_type=&q=stock)里面的paper 获得点灵感 这里面的都是有实现的
**[Stock Market Prediction via Deep Learning Techniques: A Survey](https://arxiv.org/pdf/2212.12717.pdf)**


# Papaer Summery
1. [TEA](https://downloads.hindawi.com/journals/complexity/2022/7739087.pdf)
   - Bert Twitter + LSTM
   - 认为小的time window 更好用 5 个工作日。且bert更好的embeding有很大的作用。
   - **没有 code**
2. [StockNet](https://aclanthology.org/P18-1183/)
  - 感觉 motivation 就是trend 有随机性， 但是他说有multi-tasks learning 的那部分我没有看懂。
  - Market information Encoder(MIE) Bi-GRUs + Attention
    - encode Twitter and Price
  - Variational Movement Decoder
    - real up or down (y) is depened on latern varibale(Z). VAE 
  - Attentive Temporal Auxiliary (ATA), which integrates temporal losses through an attention mechanism.
3. [CO-CPC](http://staff.ustc.edu.cn/~qiliuql/files/Publications/Guifeng-Wang-AAAI21.pdf)
   - 宏观经济指标 哪个好哥哥姐姐可以看看这个
# Propose Method
- 目前想法是按照 TEA 的 idea + Variational Movement Decoder 实验一下。 如果CO-CPC 好用的话可以加入。
- 另 features中还有
  1. Technical analysis tools. Technical analysis tools, commonly employed in traditional stock analysis, have a strong correlation with stock market performance. These tools take into account factors such as exchange rate, book-market ratio, trading volume, and other relevant financial indicators.
  2. Macroeconomic data. Macroeconomic data reflects the economic status of a specific region. Two commonly used indicators that are linked to the stock market are the Consumer Price Index (CPI) and Gross Domestic Product
  3. 感觉有难度: Graph. The industrial knowledge graph is widely utilized, not only for displaying direct connections between corporations but also for uncovering internal relationships such as upstream and downstream supply chains.


# TODO
- [x] stock data collection
- [ ] news data collection
- [ ] news data processing
- [ ] paper reading and selection
  - [ ] [ADD: Augmented Disentanglement Distillation Framework for Improving Stock Trend Forecasting](https://paperswithcode.com/paper/add-augmented-disentanglement-distillation)
  - [ ] 
- [ ] Arima hybrid model
