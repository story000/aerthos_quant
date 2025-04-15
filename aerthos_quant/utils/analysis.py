import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import shutil
import os
import openai
import litellm
import requests
from bs4 import BeautifulSoup
from aerthos_quant.api.supabase import fetch_table

def calculate_mean_gap(predictions_file, eua_file, output_file):
    in_put = pd.read_csv(predictions_file)

    # remove all the spaces in the column names
    in_put.columns = in_put.columns.str.replace(' ', '')

    in_put.reindex(range(0, len(in_put)))

    EUA_price = pd.read_csv(eua_file)

    EUA_price['Price'] = EUA_price['Price'] / EUA_price['Price'][1687]
    x_price = EUA_price[EUA_price['Date'] == '07/12/2024']['Price'].values[0] / 7.040609

    gaps = pd.DataFrame()

    # Calculate the gap between the predicted and actual values of each day
    gaps['Date'] = in_put['Date']
    for i in range(1, 11):
        gaps[f'Day_{i}'] = (in_put[f'Day_{i}_Prediction'] - in_put[f'Day_{i}_True']) / in_put[f'Day_{i}_True']

    means = []
    for i in range(1, 11):
        means.append(gaps['Day_' + str(i)].mean())

    fig, ax1 = plt.subplots(1, 3, figsize=(18, 6))  # 创建3个子图

    # 第一个图：Mean Gap
    ax1[0].plot(range(1, 11), means, color='b')
    ax1[0].set_xlabel('Days Ahead')
    ax1[0].set_ylabel('Mean Gap', color='b')
    ax1[0].set_title('Mean Gap vs Days Ahead')

    # 计算标准差
    std = []
    for i in range(1, 11):
        std.append(gaps['Day_' + str(i)].std())

    # 第二个图：Standard Deviation
    ax1[1].plot(range(1, 11), std, color='r')
    ax1[1].set_xlabel('Days Ahead')
    ax1[1].set_ylabel('Standard Deviation', color='r')
    ax1[1].set_title('Standard Deviation vs Days Ahead')
    
    # 计算置信区间
    upper_95 = []
    lower_95 = []
    upper_80 = []
    lower_80 = []
    for i in range(1, 11):
        upper_95.append(means[i-1] + 1.96 * std[i-1])
        lower_95.append(means[i-1] - 1.96 * std[i-1])
        upper_80.append(means[i-1] + 1.28 * std[i-1])
        lower_80.append(means[i-1] - 1.28 * std[i-1])

    # 第三个图：Gap of prediction
    ax1[2].plot(range(1, 11), means, label='Mean Gap')
    ax1[2].fill_between(range(1, 11), upper_95, lower_95, color='lightpink', alpha=0.5, label='95% confidence interval')
    ax1[2].fill_between(range(1, 11), upper_80, lower_80, color='violet', alpha=0.1, label='80% confidence interval')
    ax1[2].set_xlabel('Days Ahead')
    ax1[2].set_xticks(range(1, 11))
    ax1[2].set_ylabel('Gap of prediction')
    vals = ax1[2].get_yticks()
    ax1[2].set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    ax1[2].set_title('Gap of prediction vs Days Ahead')
    ax1[2].legend()
    
    plt.tight_layout()  # 确保所有图形都能适应

    plt.savefig(output_file)  
    plt.close()  

def calculate_trend(future_predictions):
    start_price = future_predictions[0]
    end_price = future_predictions[-1]
    price_change_factor = ((end_price - start_price) / start_price)*100

    max_high = max(future_predictions)
    min_low = min(future_predictions)
    average_price = (np.mean(future_predictions) + np.mean(future_predictions)) / 2
    volatility_factor = ((max_high - min_low) / average_price)*100
    
    trend_score = 50+((0.8 * price_change_factor) + (0.2 * volatility_factor))*10
    print(trend_score)

    normalized_trend_score = int(max(0, min(100, trend_score)))
    return normalized_trend_score

def calculate_sentiment(data):
    price_change_factor = float(data['Change %'].iloc[-1].replace('%', ''))/100 if not data['Change %'].isnull().all() else 0
    volume_rank = data['Vol.'].rolling(window=10).apply(lambda x: x.rank().iloc[-1], raw=False).iloc[-1]
    volume_factor = volume_rank / 10 - 0.5

    volatility_factor = (data['High'].iloc[-1] - data['Low'].iloc[-1]) / data['Open'].iloc[-1]

    sentiment_score = int(50 + (0.5 * price_change_factor + 0.3 * volume_factor + 0.2 * volatility_factor) * 50)
    return sentiment_score

def plot_actual_vs_predicted(predictions_file, output_file, x=0):
    in_put = pd.read_csv(predictions_file)
    predictions = in_put[['Date'] + [f'Day_{i}_Prediction' for i in range(1, 11)]].rename(columns=lambda x: x.replace('_Prediction', '')).T
    predictions = predictions.rename(columns=predictions.iloc[0])
    predictions.drop(predictions.index[0], inplace=True)

    # 计算预测的标准差并放入新的数据框
    std_predictions = predictions.std(axis=0)

    actuals = in_put[['Date'] + [f'Day_{i}_True' for i in range(1, 11)]].T
    actuals = actuals.rename(columns=actuals.iloc[0])
    actuals.drop(actuals.index[0], inplace=True)

    dates = in_put['Date']

    # 将日期格式化为 MM/DD/YYYY
    Dates = pd.to_datetime(dates)
    Dates = Dates.dt.strftime('%m/%d/%Y')

    upper_90_pred = pd.to_numeric(predictions[dates[x]] + 1.645 * std_predictions[dates[x]], errors='coerce')
    lower_90_pred = pd.to_numeric(predictions[dates[x]] - 1.645 * std_predictions[dates[x]], errors='coerce')
    
    figure, ax = plt.subplots(figsize=(15, 6))  # 创建一个子图
    ax.plot(range(len(actuals[dates[x]])), actuals[dates[x]].tolist(), label='Actual')
    ax.plot(range(len(predictions[dates[x]])), predictions[dates[x]].tolist(), label='Predicted')
    ax.fill_between(predictions.index, lower_90_pred, upper_90_pred, color='lightpink', alpha=0.5, label='90% confidence interval')
    ax.set_xlabel('Days Ahead')
    ax.set_ylabel('Price')
    title = 'Actual vs Predicted Price for ' + str(dates[x])
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()  # 确保所有图形都能适应
    plt.savefig(output_file)
    plt.close()
    
def trade_decisions(price, actuals, predictions, std_predictions, dates, prudence_level, leverage, x):
    wealth = 1
    list_days = ['Day_1', 'Day_2', 'Day_3', 'Day_4', 'Day_5', 'Day_6', 'Day_7', 'Day_8', 'Day_9', 'Day_10']
    position = 0
    decision = 'No move'
    current_price = price[price['Date']==Dates[x]]['Price'].values[0]
    if current_price > min(predictions[dates[x]] + norm.ppf(1 - (1 - prudence_level) / 2)*std_predictions[dates[x]]):
        decision = 'Sell'
        position = -leverage
        tp = []
        prudence_levels = [prudence_level]
        for i in range(0, 10):
            prudence_levels.append(prudence_level-(prudence_level/10*i))
        for i in range(0, 10):
            tp.append(predictions[dates[x]][list_days[i]] - norm.ppf(1 - (1 - prudence_levels[i] )/ 2)*std_predictions[dates[x]])
    elif current_price < max(predictions[dates[x]] - norm.ppf(1 - (1 - prudence_level) / 2)*std_predictions[dates[x]]):
        decision = 'Buy'
        position = leverage
        tp = []
        prudence_levels = [prudence_level]
        for i in range(0, 10):
            prudence_levels.append(prudence_level-(prudence_level/10*i))
        for i in range(0, 10):
            tp.append(predictions[dates[x]][list_days[i]] + norm.ppf(1 - (1 - prudence_levels[i] )/ 2)*std_predictions[dates[x]])
    else:
        pass

    if position == -leverage:
        for t in range(0, 10):
            if actuals[dates[x]][list_days[t]] > current_price*(1+1/leverage):
                position = 0
                wealth = 0
                break
            if actuals[dates[x]][list_days[t]] < tp[t]:
                wealth = -position + position/current_price*actuals[dates[x]][list_days[t]] + 1
                position = 0
                break
            if t == 9:
                wealth = -position + position/current_price*actuals[dates[x]][list_days[t]] + 1
                position = 0
    else:
        pass    
    
    if position == leverage:
        for t in range(0, 10):
            if actuals[dates[x]][list_days[t]] < current_price*(1-1/leverage):
                position = 0
                wealth = 0
                break
            if actuals[dates[x]][list_days[t]] > tp[t]:
                wealth = -position + position/current_price*actuals[dates[x]][list_days[t]] + 1
                position = 0
                break
            if t == 9:
                wealth = -position + position/current_price*actuals[dates[x]][list_days[t]] + 1
                position = 0

    r = wealth - 1
    try:
        return r, list_days[t], decision
    except:
        return r, 'None', decision
    
def create_gauge_chart(ax, title,value):
    # Create the gauge chart background
    ax.set_theta_offset(np.pi)  # Rotate to start from left
    ax.set_theta_direction(-1)  # Reverse the direction to make it semi-circular

    # Define each section
    sections = ['Strong Sell', 'Sell', 'Neutral', 'Buy', 'Strong Buy']
    colors = ['#FFB6C1', '#DDA0DD', '#BA55D3', '#9370DB', '#87CEFA']
    sizes = [20, 20, 20, 20, 20]  # Total should sum to 100 or full circle

    # Plot each section as a wedge
    start = 0
    for size, color in zip(sizes, colors):
        ax.barh(1, np.deg2rad(size * 1.8), left=np.deg2rad(start * 1.8), color=color, edgecolor='w', height=0.4)
        start += size

    # Add the needle to indicate the 'value'
    needle_angle = np.deg2rad((value / 100) * 180)
    ax.plot([needle_angle, needle_angle], [0, 0.9], color='k', lw=2)
    

    # Customize chart
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_title(title, fontsize=16)
    ax.text(0.5, 0.3, sections[value // 20], ha='center', fontsize=14, transform=ax.transAxes)

def get_latest_price():
    price = fetch_table('carbon')
    price = sorted(price, key=lambda x: x['Date'], reverse=True)
    return price[0]['Price']

def create_gauge_dashboard(latest_price_file, predictions_file, output_file):
    # Example usage to create a dashboard with multiple gauges
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), subplot_kw={'projection': 'polar'})
    
    latest_data = pd.read_csv(latest_price_file)
    future_trend = pd.read_csv(predictions_file).iloc[-1].tolist()[1:]
    create_gauge_chart(axs[0], 'Current Sentiment', calculate_sentiment(latest_data))
    create_gauge_chart(axs[1], 'Future Trend', calculate_trend(future_trend))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # Remove the reference circles
    for ax in axs.flat:
        ax.spines['polar'].set_visible(False)
        ax.set_yticks([])
        ax.set_xticks([])

    plt.savefig(output_file)


def plot_predictions(file_path, output_file):
    prediction = pd.read_csv(file_path)
    new_predictions = prediction[-1:]
    new_index = pd.to_datetime(new_predictions['Date'])
    
    future_dates = []
    current_date = new_index.iloc[0]
    while len(future_dates) < 10:
        current_date += pd.Timedelta(days=1)
        if current_date.weekday() < 5:  # 0-4 are Monday to Friday
            future_dates.append(current_date.strftime('%m-%d'))
    
    new_predictions.columns = ['Date'] + future_dates
    figure, ax = plt.subplots(figsize=(10, 6))
    plt.plot(new_predictions.columns[1:].tolist(), new_predictions.iloc[0, 1:].tolist(), color='orange')
    plt.xlabel('Days Ahead')
    plt.ylabel('Predicted EUA futures price EUR/ton')
    plt.title('Predicted Value vs Days Ahead')
    plt.savefig(output_file)
    
    latest_price = get_latest_price()
    trade_decision(latest_price, new_predictions.iloc[0, 1:], new_predictions.columns[1:].tolist(), 0.95, output_file)
    #get_ai_suggestions(new_predictions.iloc[0, 1:])
    
    
def trade_decision(price, predictions, list_days, prudence_level, output_file):
    decision = 'No move'
    current_price = price
    std_predictions = predictions.std()
    if current_price > min(predictions + norm.ppf(1 - (1 - prudence_level) / 2)*std_predictions):
        decision = 'Sell'
        tp = []
        prudence_levels = [prudence_level]
        for i in range(0, 10):
            prudence_levels.append(prudence_level-(prudence_level/10*i))
        for i in range(0, 10):
            tp.append(predictions[list_days[i]] - norm.ppf(1 - (1 - prudence_levels[i] )/ 2)*std_predictions)
    elif current_price < max(predictions - norm.ppf(1 - (1 - prudence_level) / 2)*std_predictions):
        decision = 'Buy'
        tp = []
        prudence_levels = [prudence_level]
        for i in range(0, 10):
            prudence_levels.append(prudence_level-(prudence_level/10*i))
        for i in range(0, 10):
            tp.append(predictions[list_days[i]] + norm.ppf(1 - (1 - prudence_levels[i] )/ 2)*std_predictions)
    else:
        print("Current price is between the confidence interval")
        print(current_price)
        print("Lower: ", min(predictions + norm.ppf(1 - (1 - prudence_level) / 2)*std_predictions))
        print("Upper: ", max(predictions - norm.ppf(1 - (1 - prudence_level) / 2)*std_predictions))
        pass

    lower = pd.to_numeric(predictions - 1.645*std_predictions, errors='coerce')
    upper = pd.to_numeric(predictions + 1.645*std_predictions, errors='coerce')

    # Visualize the predictions and the target prices (tp)
    plt.figure(figsize=(10, 6))
    plt.plot(list_days, predictions.tolist(), label='Predictions', color='violet')
    plt.fill_between(list_days, lower, upper, color='lightpink', alpha=0.5, label='90% confidence interval')
    
    
    plt.axhline(y=current_price, color='blue', linestyle='--', label='Current Price')
    plt.xlabel('Days Ahead')
    plt.ylabel('Price EUR/ton')
    plt.title('Predictions vs Days Ahead')
    plt.legend()
    plt.savefig(output_file)
    
    try: 
        return decision, tp
    except Exception as e:
        return decision, None


def get_ai_suggestions(predictions):
    history_price = get_latest_price()

    client = openai.OpenAI(
        api_key="sk--u59udu5B3LF6Wheot0vXA",
        base_url="https://cmu.litellm.ai",
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "I'm a trading advisor of carbon allowances, and I want to give my clients suggestions to buy or sell. The current price is " + str(history_price) + " and the predictions are " + str(predictions)+". Based on the predictions, analyze the short term and long term trend and give three topics, i.e. short term trend, long term trend, and the trading suggestions. Responses should be concise and to the point. No more than 150 words.",
            }
        ],
        model="gpt-4o",
    )

    print(chat_completion.choices[0].message.content)
    with open('backend/result/suggestions.txt', 'w') as f:
        f.write(chat_completion.choices[0].message.content)
        
def generate_signals(df):
    # 1. Composite Indicator (defined as a weighted average of price and volume)
    df['Composite_Indicator'] = 0.5 * df['Price'] + 0.5 * df['Vol.'] / df['Vol.'].max()

    # 2. Short Term Indicators: 7-Day Average Directional Indicator (simple 7-day moving average)
    df['7_Day_ADI'] = df['Price'].rolling(window=7).mean()

    # 3. 10-8 Day Moving Average Hilo Channel
    df['10_Day_High'] = df['High'].rolling(window=10).max()
    df['8_Day_Low'] = df['Low'].rolling(window=8).min()

    # 4. 20-Day Moving Average vs Price
    df['20_Day_MA'] = df['Price'].rolling(window=20).mean()
    df['20_Day_MA_vs_Price'] = df['Price'] - df['20_Day_MA']

    # 5. 20-50 Day MA Crossover
    df['50_Day_MA'] = df['Price'].rolling(window=50).mean()
    df['20-50_MA_Crossover'] = df['20_Day_MA'] - df['50_Day_MA']

    # 6. 20-Day Bollinger Bands
    df['20_Day_StdDev'] = df['Price'].rolling(window=20).std()
    df['Bollinger_Upper'] = df['20_Day_MA'] + (2 * df['20_Day_StdDev'])
    df['Bollinger_Lower'] = df['20_Day_MA'] - (2 * df['20_Day_StdDev'])
    
    # 1. Composite Indicator signal
    composite_avg = df['Composite_Indicator'].mean()
    df['Composite_Signal'] = 'Hold'
    df.loc[df['Composite_Indicator'] < composite_avg * 0.9, 'Composite_Signal'] = 'Buy'
    df.loc[df['Composite_Indicator'] > composite_avg * 1.1, 'Composite_Signal'] = 'Sell'

    # 2. 7-Day ADI signal
    df['7_Day_ADI_Signal'] = 'Hold'
    df.loc[(df['Price'] > df['7_Day_ADI']) & (df['7_Day_ADI'].diff() > 0), '7_Day_ADI_Signal'] = 'Buy'
    df.loc[(df['Price'] < df['7_Day_ADI']) & (df['7_Day_ADI'].diff() < 0), '7_Day_ADI_Signal'] = 'Sell'

    # 3. 10-8 Day Hilo Channel signal
    df['10_8_Day_Hilo_Channel_Signal'] = 'Hold'
    df.loc[df['Price'] > df['10_Day_High'], '10_8_Day_Hilo_Channel_Signal'] = 'Buy'
    df.loc[df['Price'] < df['8_Day_Low'], '10_8_Day_Hilo_Channel_Signal'] = 'Sell'

    # 4. 20-Day MA vs Price signal
    df['20_Day_MA_vs_Price_Signal'] = 'Hold'
    df.loc[df['Price'] > df['20_Day_MA'], '20_Day_MA_vs_Price_Signal'] = 'Buy'
    df.loc[df['Price'] < df['20_Day_MA'], '20_Day_MA_vs_Price_Signal'] = 'Sell'

    # 5. 20-50 MA Crossover signal
    df['20-50_MA_Crossover_Signal'] = 'Hold'
    df.loc[df['20-50_MA_Crossover'] > 0, '20-50_MA_Crossover_Signal'] = 'Buy'
    df.loc[df['20-50_MA_Crossover'] < 0, '20-50_MA_Crossover_Signal'] = 'Sell'

    # 6. Bollinger Bands signal
    df['Bollinger_Bands_Signal'] = 'Hold'
    df.loc[df['Price'] < df['Bollinger_Lower'], 'Bollinger_Bands_Signal'] = 'Buy'
    df.loc[df['Price'] > df['Bollinger_Upper'], 'Bollinger_Bands_Signal'] = 'Sell'

    # Return final signals
    return df[['Date', 'Composite_Signal', '7_Day_ADI_Signal', '10_8_Day_Hilo_Channel_Signal', '20_Day_MA_vs_Price_Signal', '20-50_MA_Crossover_Signal', 'Bollinger_Bands_Signal']]

def get_opinions(output_file):
    

    url = "https://carbonherald.com/opinion/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36'
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        articles = soup.find_all('article')  # Assume the title is in the <h2> tag
        
        articles_list = []
        for article in articles:
            if 'Opinion' in article.get_text():
                article_info = dict()
                text = article.get_text().split('Opinion')[-1]
                a_tag = article.find('a')  # Search for the <a> tag
                article_info['link'] = a_tag['href']
                article_info['title'] = a_tag['href'].replace('/', '').split('.com')[-1].replace('-', ' ').title()
                author_start = text.split('…by')[-1]
                months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
                for month in months:
                    if month in author_start:
                        article_info['author'] = author_start.split(month)[0].strip()
                        article_info['Date'] = month + ' ' + author_start.split(month)[-1].strip()
                        break
                article_info['text'] = text
                
            
                
                articles_list.append(article_info)
      
    else:
        print("无法访问网页，状态码:", response.status_code)
    df = pd.read_csv(output_file)
    df_new = pd.DataFrame(articles_list)
    df = pd.concat([df, df_new]).drop_duplicates().reset_index(drop=True)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')



#create_gauge_dashboard('backend/data/latest_price.csv', 'backend/data/predictions.csv', 'backend/result/indicators_dashboard.png')

plot_predictions('data/processed/predictions.csv', 'aerthos_quant/predictions/predictions_plot.png')

# get_opinions('backend/result/opinions.csv')

# signals = generate_signals(pd.read_csv('backend/data/latest_price.csv'))
# signals.to_csv('backend/result/signals.csv', index=False)
