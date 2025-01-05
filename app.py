import streamlit as st
import pandas as pd
import altair as alt

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

with st.expander("Show a Sample of the Scraped Dataset from r/CryptoCurrency"):
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

    with st.expander("Scores for all labels:"): 
        st.dataframe(scores_df)

    # Score development chart
    st.header("Time-Series Score Visualization")
    adjustment_option = st.selectbox("Select Score Type:", ["Popularity Score", "Sentiment Score"])
    include_price = st.checkbox("Include Daily Price Development [in Red]") 

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

        if include_price:
            daily_prices['date'] = pd.to_datetime(daily_prices['date'], errors='coerce')
            coin_column = f"{selected_label}_price"
            if coin_column in daily_prices.columns:
                merged_data = pd.merge(score_development, daily_prices[['date', coin_column]], on='date', how='inner')
                merged_data.rename(columns={coin_column: 'price'}, inplace=True)

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

    #Time-Series score visualization
    st.header("Gradient Score Dynamic")
    visualization_option = st.selectbox("Select Gradient Score Type:", ["Popularity Score", "Sentiment Score"])

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

        #merge with daily prices
        coin_column = f"{selected_label}_price"
        if coin_column in daily_prices.columns:
            merged_data = pd.merge(selected_time_scores, daily_prices[['date', coin_column]], on='date', how='inner')
            merged_data.rename(columns={coin_column: 'price'}, inplace=True)

            #calculate gradients for score and price
            merged_data['Score Gradient'] = merged_data['score'].diff()
            merged_data['Price Gradient'] = merged_data['price'].diff()

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

st.sidebar.header("Lagged Correlation")

if 'Score Gradient' in merged_data.columns and 'Price Gradient' in merged_data.columns:

    correlation = merged_data[['Score Gradient', 'Price Gradient']].corr().iloc[0, 1]

    # lagged correlations
    lag_days = 1  #change this value to calculate lag correlations
    merged_data['Lagged Score Gradient'] = merged_data['Score Gradient'].shift(lag_days)
    lagged_correlation = merged_data[['Lagged Score Gradient', 'Price Gradient']].corr().iloc[0, 1]

    merged_data['Lagged Price Gradient'] = merged_data['Price Gradient'].shift(lag_days)
    lagged_price_correlation = merged_data[['Score Gradient', 'Lagged Price Gradient']].corr().iloc[0, 1]


    st.sidebar.write(f"Score-Leads-Price-Corr. = **{lagged_correlation:.2f}**")
    st.sidebar.write(f"Price-Leads-Score-Corr. = **{lagged_price_correlation:.2f}**")

st.sidebar.header("Result:")

# Predictive strength interpretation 
if lagged_correlation > lagged_price_correlation:
    if lagged_correlation < 0:
        st.sidebar.markdown("<span style='color:green'>**There is an inverse relationship: Reddit score changes tend to move opposite to price changes.**</span>", unsafe_allow_html=True)
    elif lagged_correlation < 0.25:
        st.sidebar.markdown("<span style='color:green'>**There is a small chance to predict the price by Reddit score.**</span>", unsafe_allow_html=True)
    elif lagged_correlation < 0.75:
        st.sidebar.markdown("<span style='color:green'>**There is a significant chance to predict the price by Reddit score.**</span>", unsafe_allow_html=True)
    else:
        st.sidebar.markdown("<span style='color:green'>**There is a high chance to predict the price by Reddit score.**</span>", unsafe_allow_html=True)
elif lagged_correlation < lagged_price_correlation:
    st.sidebar.markdown("<span style='color:red'>**No price prediction possible. Reddit score reflects price changes.**</span>", unsafe_allow_html=True)
else:
    st.sidebar.markdown("**The correlation is balanced, showing no clear lead between Reddit score and price changes.**")

