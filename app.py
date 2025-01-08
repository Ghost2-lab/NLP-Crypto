import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

data_path = 'r_CryptoCurrency_classified_compressed.csv'
price_data_path = 'daily_prices_dec_2024.csv'
sample_path = 'r_CryptoCurrency_classified_sample.csv'
data = pd.read_csv(data_path)
daily_prices = pd.read_csv(price_data_path)
sample = pd.read_csv(sample_path)

if 'created_utc' in data.columns:
    data['created_utc'] = pd.to_datetime(data['created_utc'], dayfirst=True, errors='coerce')

if 'date' in daily_prices.columns:
    daily_prices['date'] = pd.to_datetime(daily_prices['date'], dayfirst=True, errors='coerce')

st.title("Predicting Cryptocurrency Trends and Prices Using Reddit Submissions")
st.markdown("<span style='color:red'>**This is a demo version of the app and contains only a sample of submissions from r/CryptoCurrency, specifically from December 2024. In the full version, the app will automatically extract new submissions and run in real-time, enabling continuous monitoring of subreddit sentiment.**</span>", unsafe_allow_html=True)

with st.expander("Expand to read more important informations:"):
    st.markdown("#### Root Hypothesis")
    st.write("The project was inspired by the debate on whether cryptocurrency prices are driven solely by technological factors, such as safety and functionality, or are also significantly influenced by social dynamics. For instance, could price changes be largely fueled by hype created and amplified on social media platforms?")
    st.write("Therefore, the hypothesis can be stated as follows: Cryptocurrency price movements are significantly influenced by social media discussions, where sentiment and hype play a major role.")
    st.markdown("#### Research Objective")
    st.write("The primary objective is to evaluate whether Reddit posts, comments, and votes can be used to calculate a popularity score that correlates with cryptocurrency price trends. This research seeks to understand the potential of sentiment analysis as a predictive tool for cryptocurrency trends while acknowledging the possibility that no significant correlation might be found. The project places emphasis on the methodological approach rather than the success of correlation detection.")
    st.markdown("#### Methodolgy")
    st.write("Data Collection - Data will be gathered from the r/CryptoCurrency subreddit using the Reddit API. To ensure the process remains efficient and relevant, data collection will be limited to a specific time range, reducing computational demands while maintaining the scope's manageability.")
    st.write("Processing and Sentiment Analysis - The dataset will be prepared with labels corresponding to each cryptocurrency. Sentiment analysis will be conducted using a pre-trained model as a baseline, with potential fine-tuning to adapt to the unique linguistic patterns and context of cryptocurrency discussions on Reddit. If needed, a portion of the data will be manually labeled to enhance model accuracy during training and evaluation.")
    st.write("Correlation Assessment - The project will assess whether sentiment scores demonstrate a meaningful correlation with cryptocurrency price trends, identifying potential predictive relationships and offering insights into the dynamics between social sentiment and market movements.")
    st.markdown("#### Sample of the Scraped Dataset from r/CryptoCurrency")
    st.dataframe(sample.sample(n=10))

st.header("Trending Cryptocurrencies on Reddit")


if 'labeled_submission' in data.columns and 'comment_score' in data.columns:
    unique_labels = data['labeled_submission'].unique()  # Get all unique labels
    selected_label = st.sidebar.selectbox("Select a Crypto for time-series visualization:", unique_labels)

    date_min = pd.Timestamp('2024-12-01')
    date_max = pd.Timestamp('2024-12-31')
    selected_date_range = st.sidebar.date_input("Select a date range:", [date_min, date_max], min_value=date_min, max_value=date_max)


    if len(selected_date_range) == 2:
        start_date, end_date = pd.Timestamp(selected_date_range[0]), pd.Timestamp(selected_date_range[1])
    else:
        start_date, end_date = date_min, date_max

    # Calculate scores
    scores = {
        label: data[(data['labeled_submission'] == label) & (data['created_utc'] >= start_date) & (data['created_utc'] <= end_date)]['comment_score'].sum() for label in unique_labels
    }
    scores_df = pd.DataFrame(list(scores.items()), columns=['Label', 'Score']).sort_values(by='Score', ascending=False)

    top_10_scores_df = scores_df.head(10)

    top_10_chart = alt.Chart(top_10_scores_df).mark_bar().encode(
        x=alt.X('Label:N', sort='-y', title="Cryptocurrencies Symbol"),
        y=alt.Y('Score:Q', title="Raw Score"),
        tooltip=['Label', 'Score']
    ).properties(
        title="Top 10 Cryptocurrencies by Popularity Score in the Selected Time Range"
    )

    st.altair_chart(top_10_chart, use_container_width=True)

    with st.expander("Expand to read explanation:"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("Scores for all labels")
            st.dataframe(scores_df)
        with col2:
            st.write("The displayed graph and table showcase the cryptocurrencies and their corresponding raw Reddit scores accumulated during the specified time period. These scores indicate which cryptocurrencies are most frequently discussed within the subreddit r/CryptoCurrency. This analysis provides insights into the popularity and trends of various cryptocurrencies. If implemented in real-time, such data could effectively monitor which cryptocurrencies are gaining hype within the community.")
            st.write("Additionally, tracking the raw Reddit scores over time allows for the identification of emerging patterns and sudden spikes in discussions. These trends could signal increased interest or hype, potentially reflecting upcoming market movements.")
# Score development chart
st.header("Time-Series Score Visualization")
adjustment_option = st.selectbox("Select Score Type:", ["Popularity Score", "Sentiment Score"], index=1)
include_price = st.checkbox("Include Daily Price Development [in Red]", value=True) 

if 'created_utc' in data.columns:
    if adjustment_option == "Popularity Score":
        score_development = data[(data['labeled_submission'] == selected_label) & (data['created_utc'] >= start_date) & (data['created_utc'] <= end_date)]
        score_development = score_development.groupby(score_development['created_utc'].dt.date)['comment_score'].sum().reset_index()
        score_development.columns = ['date', 'score']
    else:
        score_development = data[(data['labeled_submission'] == selected_label) & (data['created_utc'] >= start_date) & (data['created_utc'] <= end_date)]
        score_development['adjusted_score'] = score_development.apply(
            lambda row: row['comment_score'] if row['Sentiment'] == 'Positive' else -row['comment_score'] if row['Sentiment'] == 'Negative' else 0,
            axis=1
        )
        score_development = score_development.groupby(score_development['created_utc'].dt.date)['adjusted_score'].sum().reset_index()
        score_development.columns = ['date', 'score']

    score_development['date'] = pd.to_datetime(score_development['date'], errors='coerce')

    x_values = (score_development['date'] - score_development['date'].min()).dt.days
    y_values = score_development['score']

    positive_area = np.trapz(y_values[y_values > 0], x=x_values[y_values > 0])
    negative_area = np.trapz(y_values[y_values < 0], x=x_values[y_values < 0])

    total_area = abs(positive_area) + abs(negative_area)
    positive_percentage = (abs(positive_area) / total_area) * 100 if total_area != 0 else 0
    negative_percentage = (abs(negative_area) / total_area) * 100 if total_area != 0 else 0

    score_chart = alt.Chart(score_development).mark_line().encode(
        x=alt.X('date:T', title="Time"),
        y=alt.Y('score:Q', title="Score"),
        tooltip=['date:T', 'score:Q']
    ).properties(
        title=f"Score Development for {selected_label} ({adjustment_option})"
    )

    if include_price:
        daily_prices['date'] = pd.to_datetime(daily_prices['date'], errors='coerce')
        coin_column = f"{selected_label}_price"
        if coin_column in daily_prices.columns:
            merged_data = pd.merge(score_development, daily_prices[['date', coin_column]], on='date', how='inner')
            merged_data.rename(columns={coin_column: 'price'}, inplace=True)

            # Calculate price development
            price_first = merged_data['price'].iloc[0]
            price_latest = merged_data['price'].iloc[-1]
            price_development = ((price_latest - price_first) / price_first) * 100 if price_first != 0 else 0

            # Altair chart 
            score_chart = alt.Chart(merged_data).mark_line().encode(
                x=alt.X('date:T', title="Time"),
                y=alt.Y('score:Q', title="Score"),
                tooltip=['date:T', 'score:Q']
            ).properties(
                title=f"Score Development for {selected_label} ({adjustment_option})"
            )

            price_chart = alt.Chart(merged_data).mark_line(color='red').encode(
                x=alt.X('date:T', title="Time"),
                y=alt.Y('price:Q', title="Price"),
                tooltip=['date:T', 'price:Q']
            ).properties(
                title=f"Daily Price Development for {selected_label}"
            )

            combined_chart = alt.layer(score_chart, price_chart).resolve_scale(
                y='independent'
            )
            st.altair_chart(combined_chart, use_container_width=True)
        else:
            st.error(f"Price data for {selected_label} is not available in the daily prices dataset.")
    else:
        score_chart = alt.Chart(score_development).mark_line().encode(
            x=alt.X('date:T', title="Time"),
            y=alt.Y('score:Q', title="Score"),
            tooltip=['date:T', 'score:Q']
        ).properties(
            title=f"Score Development for {selected_label} ({adjustment_option})"
        )

        # Display score chart
        st.altair_chart(score_chart, use_container_width=True)


    with st.expander("Expand to read graph description:"):
        st.write("This section generates a time-series visualization that analyzes and displays the development of Reddit scores for a selected cryptocurrency over a user-specified time range. It provides insights into how the discussion sentiment or popularity on Reddit evolves over time and optionally overlays cryptocurrency price trends for comparison.")
        # st.markdown(f"Positive Sentiment: {positive_percentage:.2f}%")
        # st.markdown(f"Negative Sentiment: {negative_percentage:.2f}%")
           
            
# Time-Series score visualization
st.header("Gradient Score Dynamic")
visualization_option = st.selectbox("Select Gradient Score Type:", ["Popularity Score", "Sentiment Score"], index=1)

if 'created_utc' in data.columns:
    if visualization_option == "Popularity Score":
        selected_time_scores = data[(data['labeled_submission'] == selected_label) & (data['created_utc'] >= start_date) & (data['created_utc'] <= end_date)]
        selected_time_scores = selected_time_scores.groupby(selected_time_scores['created_utc'].dt.date)['comment_score'].sum().reset_index()
    else:
        selected_time_scores = data[(data['labeled_submission'] == selected_label) & (data['created_utc'] >= start_date) & (data['created_utc'] <= end_date)]
        selected_time_scores['adjusted_score'] = selected_time_scores.apply(
            lambda row: row['comment_score'] if row['Sentiment'] == 'Positive' else -row['comment_score'] if row['Sentiment'] == 'Negative' else 0,
            axis=1
        )
        selected_time_scores = selected_time_scores.groupby(selected_time_scores['created_utc'].dt.date)['adjusted_score'].sum().reset_index()

    selected_time_scores.columns = ['date', 'score']

    selected_time_scores['date'] = pd.to_datetime(selected_time_scores['date'], errors='coerce')
    daily_prices['date'] = pd.to_datetime(daily_prices['date'], errors='coerce')

    # Merge with daily prices
    coin_column = f"{selected_label}_price"
    if coin_column in daily_prices.columns:
        merged_data = pd.merge(selected_time_scores, daily_prices[['date', coin_column]], on='date', how='inner')
        merged_data.rename(columns={coin_column: 'price'}, inplace=True)

        # Calculate gradients for score and price
        merged_data['Score Gradient'] = merged_data['score'].diff()
        merged_data['Price Gradient'] = merged_data['price'].diff()

        x_values_gradient = (merged_data['date'] - merged_data['date'].min()).dt.days
        y_values_gradient_score = merged_data['Score Gradient']
        y_values_gradient_price = merged_data['Price Gradient']

        # Calculate areas for score gradients
        positive_area_gradient_score = np.trapz(y_values_gradient_score[y_values_gradient_score > 0], x=x_values_gradient[y_values_gradient_score > 0])
        negative_area_gradient_score = np.trapz(y_values_gradient_score[y_values_gradient_score < 0], x=x_values_gradient[y_values_gradient_score < 0])

        total_area_gradient_score = abs(positive_area_gradient_score) + abs(negative_area_gradient_score)
        positive_percentage_gradient_score = (abs(positive_area_gradient_score) / total_area_gradient_score) * 100 if total_area_gradient_score != 0 else 0
        negative_percentage_gradient_score = (abs(negative_area_gradient_score) / total_area_gradient_score) * 100 if total_area_gradient_score != 0 else 0

        # Calculate areas for price gradients
        positive_area_gradient_price = np.trapz(y_values_gradient_price[y_values_gradient_price > 0], x=x_values_gradient[y_values_gradient_price > 0])
        negative_area_gradient_price = np.trapz(y_values_gradient_price[y_values_gradient_price < 0], x=x_values_gradient[y_values_gradient_price < 0])

        total_area_gradient_price = abs(positive_area_gradient_price) + abs(negative_area_gradient_price)
        positive_percentage_gradient_price = (abs(positive_area_gradient_price) / total_area_gradient_price) * 100 if total_area_gradient_price != 0 else 0
        negative_percentage_gradient_price = (abs(negative_area_gradient_price) / total_area_gradient_price) * 100 if total_area_gradient_price != 0 else 0

        gradient_chart = alt.Chart(merged_data).mark_line().encode(
            x=alt.X('date:T', title="Time"),
            y=alt.Y('Score Gradient:Q', title="Score Gradient"),
            tooltip=['date:T', 'Score Gradient:Q']
        ).properties(
            title=f"Gradient of Scores for {selected_label} ({visualization_option})"
        )

        price_gradient_chart = alt.Chart(merged_data).mark_line(color='red').encode(
            x=alt.X('date:T', title="Time"),
            y=alt.Y('Price Gradient:Q', title="Price Gradient"),
            tooltip=['date:T', 'Price Gradient:Q']
        ).properties(
            title=f"Gradient of Prices for {selected_label}"
        )

        combined_gradient_chart = alt.layer(gradient_chart, price_gradient_chart).resolve_scale(
            y='independent'
        )

        st.altair_chart(combined_gradient_chart, use_container_width=True)

        

  
            
    else:
        st.error(f"Price data for {selected_label} is not available in the daily prices dataset.")
else:
    st.error("The necessary columns ('labeled_submission', 'comment_score', and 'created_utc') do not exist in the dataset.")


    # st.write(f"For the given time period {selected_label} experience sentiment GROWTH {positive_percentage_gradient:.2f}% of the time")
    # st.write(f"Decline: {negative_percentage_gradient:.2f}%")

    # with st.expander("Expand to read future research potential:"):
    #     st.write(f"Growth: {positive_percentage_gradient:.2f}%")
    #     st.write(f"Decline: {negative_percentage_gradient:.2f}%")







if 'Score Gradient' in merged_data.columns and 'Price Gradient' in merged_data.columns:

    correlation = merged_data[['Score Gradient', 'Price Gradient']].corr().iloc[0, 1]

    # lagged correlations
    lag_days = 1  #change this value to calculate lag correlations
    merged_data['Lagged Score Gradient'] = merged_data['Score Gradient'].shift(lag_days)
    lagged_correlation = merged_data[['Lagged Score Gradient', 'Price Gradient']].corr().iloc[0, 1]

    merged_data['Lagged Price Gradient'] = merged_data['Price Gradient'].shift(lag_days)
    lagged_price_correlation = merged_data[['Score Gradient', 'Lagged Price Gradient']].corr().iloc[0, 1]

with st.expander(f"Show metrics for performance and predictive analysis:"):
    st.write("The Gradient Score Dynamic section analyzes how changes (gradients) in Reddit scores and cryptocurrency prices evolve over time. It provides insights into the rate of change in discussion scores and price movements, allowing for a deeper understanding of their dynamic relationship.")
    st.write(f"- Sentiment growth {positive_percentage_gradient_score:.2f}% of the time")
    st.write(f"- Sentiment decline {negative_percentage_gradient_score:.2f}% of the time")
    st.write(f"- Price growth {positive_percentage_gradient_price:.2f}% of the time")
    st.write(f"- Price decline {negative_percentage_gradient_price:.2f}% of the time")
    st.write("Lagged Correlation")
    st.write(f"- Score-Leads-Price-Corr. = **{lagged_correlation:.2f}**")
    st.write(f"- Price-Leads-Score-Corr. = **{lagged_price_correlation:.2f}**")
    st.write("Lagged correlation measures the relationship between two variables (e.g., Reddit scores and cryptocurrency prices) at different time offsets to identify whether one precedes the other.")
    st.write("For predictive analysis, lagged correlation is valuable in determining if changes in Reddit sentiment can predict price movements or if social sentiment merely reacts to market changes. A positive lagged correlation where scores lead prices suggests predictive potential, while prices leading scores imply a reactionary relationship.")
    st.write("This analysis helps investors and researchers assess whether social media sentiment is a useful signal for forecasting market trends, though it does not imply causation and may vary across time or cryptocurrencies.")



st.header("Future Research Potential")
st.write("Additional efforts could include implementing advanced methods to filter out bot-generated content, reducing noise and ensuring higher data quality. This would help achieve a more reliable analysis of genuine user sentiment. Furthermore, exploring the impact of temporal lags between sentiment changes and market reactions could offer valuable insights into the timing and causality of these dynamics. These enhancements would improve the robustness and predictive power of the analysis.")

# Provide insights about price development and sentiment analysis
if 'positive_percentage_gradient_score' in locals() and 'negative_percentage_gradient_score' in locals() and 'price_development' in locals():
    st.sidebar.write(
        f"For the selected date range, {selected_label} showed {positive_percentage_gradient_score:.2f}% positive sentiment "
        f"and {negative_percentage_gradient_score:.2f}% negative sentiment. "
        f"Additionally, {selected_label} experienced a price development of {price_development:.2f}%."
    )
else:
    st.sidebar.write(
        f"For the selected date range, {selected_label} showed {positive_percentage_gradient_score:.2f}% positive sentiment "
        f"and {negative_percentage_gradient_score:.2f}% negative sentiment. "
    )

st.sidebar.header("Correlation Analysis")

# Predictive strength interpretation
if lagged_correlation > lagged_price_correlation:
    if lagged_correlation < 0:
        st.sidebar.markdown("<span style='color:green'>**The lagged correlation of indicates that changes in Reddit scores are inversely related to price changes, meaning that an increase in scores often leads to a decrease in prices and vice versa.**</span>", unsafe_allow_html=True)
    elif lagged_correlation < 0.25:
        st.sidebar.markdown("<span style='color:green'>**The lagged correlation of suggests that changes in Reddit scores have minimal predictive power for price changes, with only a weak influence on price movements.**</span>", unsafe_allow_html=True)
    elif lagged_correlation < 0.75:
        st.sidebar.markdown("<span style='color:green'>**The lagged correlation shows that changes in Reddit scores moderately predict price changes, suggesting a significant but not definitive influence of sentiment scores on prices.**</span>", unsafe_allow_html=True)
    else:
        st.sidebar.markdown("<span style='color:green'>**The lagged correlation indicates that changes in Reddit scores strongly predict price changes, showing a high probability that sentiment scores influence price movements effectively.**</span>", unsafe_allow_html=True)
elif lagged_correlation < lagged_price_correlation:
    st.sidebar.markdown("<span style='color:red'>**The lagged correlation suggests that price changes lead Reddit score changes, implying no predictive value in using Reddit scores for forecasting prices.**</span>", unsafe_allow_html=True)
else:
    st.sidebar.markdown("**The correlation is balanced, indicating that neither Reddit scores nor price changes have a leading influence on the other, suggesting no clear predictive relationship.**")




