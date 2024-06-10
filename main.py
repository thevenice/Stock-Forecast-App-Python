import streamlit as st
from datetime import date

# Data download
import yfinance as yf

# Forecasting
from prophet import Prophet
from prophet.plot import plot_plotly

# Data visualization
import plotly.graph_objects as go


# Set start and end dates (consider recent data)
START = "2020-01-01"  # Adjust for more recent data
TODAY = date.today().strftime("%Y-%m-%d")


def main():
    """
    Main function for the Stock Forecast App
    """

    # App title
    st.title('Stock Forecast App')

    # Stock options (update with current popular tickers)
    stocks = ('AAPL', 'TSLA', 'AMZN', 'NVDA')
    selected_stock = st.selectbox('Select stock for prediction', stocks)

    # Prediction period selection
    n_years = st.slider('Prediction horizon (years):', 1, 5)
    period = n_years * 365

    @st.cache_data  # Use recommended caching function
    def load_data(ticker):
        """Downloads and cleans stock data"""
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

    data_load_state = st.text('Loading data...')
    data = load_data(selected_stock)
    data_load_state.text('Loading data... done!')

    st.subheader('Raw Data')
    st.write(data.tail())


    def plot_raw_data():
        """Visualizes historical open and close prices"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Opening Price"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Closing Price"))
        fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)


    plot_raw_data()

    # Prepare data for Prophet
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    # Create and fit Prophet model
    m = Prophet()
    m.fit(df_train)

    # Generate future dates for prediction
    future = m.make_future_dataframe(periods=period)

    # Predict future stock prices
    forecast = m.predict(future)

    # Display forecast data
    st.subheader('Forecast Data')
    st.write(forecast.tail())

    # Forecast plot
    st.write(f'Forecast for the next {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    # Optional forecast components plot (commented out)
    # st.write("Forecast Components")
    # fig2 = m.plot_components(forecast)
    # # Forecast components plot (using Matplotlib)
    # st.write("Forecast Components")
    # st.pyplot(fig2)


if __name__ == "__main__":
    main()
