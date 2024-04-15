import streamlit as st
import pandas as pd
import requests
import streamlit_option_menu
from streamlit_option_menu import option_menu
import datetime
import asyncio
import websockets
from influxdb_client import InfluxDBClient, Point, Dialect
from influxdb_client.client.flux_table import FluxRecord
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv()

# Constants for InfluxDB connection
# url = 'https://us-east-1-1.aws.cloud2.influxdata.com'
# token = os.getenv('INFLUX_TOKEN')
# org = os.getenv('ORG_NAME')
# bucket = os.getenv('DATABASE')

# # Function to query the most recent data from InfluxDB
# async def fetch_latest_data():
#     query = f"""
#     from(bucket: "{bucket}")
#     |> range(start: -1d)
#     |> last()
#     """
#     with InfluxDBClient(url=url, token=token, org=org) as client:
#         query_api = client.query_api()
#         result = query_api.query(query=query, org=org)
#         records = []
#         for table in result:
#             for record in table.records:
#                 records.append(record)
#         if records:
#             return str(records[-1].values)
#         return "No data found."

# # Function to handle WebSocket connections
# async def websocket_handler(websocket, path):
#     while True:
#         try:
#             # Wait for any message from the client
#             message = await websocket.recv()
#             print(f"Received request: {message}")

#             # Fetch the latest data from InfluxDB
#             data = await fetch_latest_data()
#             await websocket.send(data)
#         except websockets.exceptions.ConnectionClosed:
#             print("Connection closed")
#             break

# # Start the WebSocket server
# start_server = websockets.serve(websocket_handler, "localhost", 6789)
# print("WebSocket server started on ws://localhost:6789")
# asyncio.get_event_loop().run_until_complete(start_server)
# asyncio.get_event_loop().run_forever()


# Making a get request 
responses = requests.get('http://localhost:4000/fetchData').json()
altitudes, humidity, pressure, temperature, time = [], [], [], [], []

def convert_time(time):

    seconds = time / 1000.0 # Convert milliseconds to seconds (since datetime expects seconds)
    date_time = datetime.datetime.fromtimestamp(seconds) # Create a datetime object from the seconds
    return date_time.strftime('%Y-%m-%d %H:%M:%S') # Print the datetime in a human-readable format



for response in responses:

    altitudes.append(response['altitude'])
    humidity.append(response['humidity'])
    pressure.append(response['pressure'])
    temperature.append(response['temperature'])
    time.append(convert_time(response['time']))



with st.sidebar:
    page = option_menu(
    menu_title = "Main Menu",
    options = ["Real-time", "Analytics"],
    icons = ['house', 'cloud-upload'],
    menu_icon = "cast",
    default_index = 0,
    # orientation = "horizontal",
)
    
# Setting a header for the app
st.header("Live Weather Monitoring")

if page == "Analytics":
    genre = st.radio(
        "Choose Data to Visualize",
        ["Altitude", "Humidity", "Pressure", "Temperature"],
    )

    if genre == "Altitude":
        st.header("Altitude Analytics")
        chart_data = pd.DataFrame(
                altitudes,time)
        st.line_chart(chart_data, color=["#FF0000"])

    elif genre == "Humidity":
        st.header("Humidity Analytics")
        chart_data = pd.DataFrame(
                humidity,time)
        st.line_chart(chart_data, color=["#0000FF"])
        
    elif genre == "Pressure":
        st.header("Pressure Analytics")
        chart_data = pd.DataFrame(
                pressure,time)
        st.line_chart(chart_data, color=["#00FF00"])

    elif genre == "Temperature":
        st.header("Temperature Analytics")
        chart_data = pd.DataFrame(
                temperature,time)
        st.line_chart(chart_data, color=["#FFFF00"])

