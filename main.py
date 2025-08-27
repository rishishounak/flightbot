# main.py

from flask import Flask, request, jsonify
import requests
import threading
import time
import csv

# LangChain imports
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI

app = Flask(__name__)




# --- CONFIGURATION ---
AVIATIONSTACK_KEY = ""
OPENAI_API_KEY = ""


# City → IATA mapping
city_to_iata = {
    "Delhi": "DEL",
    "Mumbai": "BOM",
    "Bangalore": "BLR"
}

# --- COMMON SERVER FUNCTION ---
@app.route("/common/iata")
def get_iata():
    city = request.args.get("city")
    iata = city_to_iata.get(city)
    return jsonify({"iata": iata})

# --- ATLAS SERVER FUNCTION (with pagination up to 3 pages) ---
@app.route("/atlas/flights")
def get_flights():
    origin = request.args.get("origin")
    all_flights = []
    page = 1
    max_pages = 2

    while page <= max_pages:
        url = f"http://api.aviationstack.com/v1/flights?access_key={AVIATIONSTACK_KEY}&dep_iata={origin}&limit=100&offset={(page-1)*100}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            try:
                data = response.json()
            except ValueError:
                print(f"JSON decode error on page {page}! Response text:", response.text)
                break
        except requests.exceptions.RequestException as e:
            print(f"Request failed on page {page}:", e)
            break

        print(f"API Response Page {page}:", data)

        flights_data = data.get("data", [])
        if not flights_data:
            break

        all_flights.extend(flights_data)
        page += 1

    # Store flights in CSV if any
    if all_flights:
        with open(f"flights_{origin}.csv", "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[
                "airline_name", "flight_iata", "departure_scheduled", "departure_airport", "arrival_airport"
            ])
            writer.writeheader()
            for f in all_flights:
                writer.writerow({
                    "airline_name": f["airline"]["name"],
                    "flight_iata": f["flight"]["iata"],
                    "departure_scheduled": f["departure"]["scheduled"],
                    "departure_airport": f["departure"]["airport"],
                    "arrival_airport": f["arrival"]["airport"]
                })
    else:
        print("No flight data retrieved.")

    return jsonify({"data": all_flights})

# --- Python Functions to Use as LangChain Tools ---
def fetch_iata(city: str) -> str:
    res = requests.get(f"http://127.0.0.1:5000/common/iata?city={city}")
    return res.json().get("iata", "")

def fetch_flights(iata: str) -> str:
    res = requests.get(f"http://127.0.0.1:5000/atlas/flights?origin={iata}")
    data = res.json()  # already parsed once in server
    flights_data = data.get("data", [])

    if not flights_data:
        return "No data retrieved."

    # Show first 20 flights in LangChain output for readability
    output = []
    for f in flights_data[:20]:
        output.append(f"{f['airline']['name']} - {f['flight']['iata']} - {f['departure']['scheduled']}")
    return "\n".join(output)

# --- Define LangChain Tools ---
tools = [
    Tool(
        name="GetIATA",
        func=fetch_iata,
        description="Get IATA code from city name"
    ),
    Tool(
        name="GetFlights",
        func=fetch_flights,
        description="Get flights departing from an IATA code"
    )
]

# --- MCP Client using LangChain Agent ---
def run_langchain_query(user_query: str) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)
    result = agent.run(user_query)
    return result

# --- FLASK QUERY ENDPOINT ---
@app.route("/query")
def query_endpoint():
    user_query = request.args.get("q")
    result = run_langchain_query(user_query)
    return jsonify({"result": result})

# --- RUN FLASK APP ---
if __name__ == "__main__":
    def run_app():
        app.run(port=5000)

    threading.Thread(target=run_app).start()

    time.sleep(2)  # wait for server to start

    # --- EXAMPLE USAGE ---
    query = "show flights from Delhi"
    print("Query:", query)
    result_text = run_langchain_query(query)
    print("\nFlights Result:\n", result_text)
