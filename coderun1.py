
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, date
import random
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# =======================
# GLOBAL CONSTANTS
# =======================
GST_RATE = 0.18

# =======================
# LOAD DATA FROM CSV FILES
# =======================
@st.cache_data
def load_data():
    try:
        flights = pd.read_csv("flight_csv.csv")
        hotels = pd.read_csv("hotels_large.csv")
        sights = pd.read_csv("sightseeing.csv")
        buses = pd.read_csv("bus_csv.csv")
        trains = pd.read_csv("train_csv.csv")
        taxis = pd.read_csv("taxi_travels.csv")
        events = pd.read_csv("events_data_updated.csv")
        return flights, hotels, sights, buses, trains, taxis, events
    except Exception as e:
        st.error(f"Error loading data: {e}")
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty, empty, empty

FLIGHTS_DF, HOTELS_DF, SIGHTS_DF, BUS_DF, TRAIN_DF, TAXI_DF, EVENTS_DF = load_data()

# =======================
# HELPER FUNCTIONS
# =======================
def calculate_gst(amount):
    return int(amount * GST_RATE)

def get_nights(check_in, check_out):
    if isinstance(check_in, date) and isinstance(check_out, date):
        return (check_out - check_in).days
    return 0

def get_column_value(row, possible_names, default=""):
    """Get value from row using multiple possible column names."""
    for name in possible_names:
        if name in row.index:
            val = row[name]
            if pd.isna(val):
                return default
            return val
    return default

def safe_int_convert(value, default=0):
    """Safely convert a value to integer, returning default if conversion fails."""
    try:
        if pd.isna(value):
            return default
        if isinstance(value, str):
            value = value.replace('₹', '').replace(',', '').strip()
        return int(float(value))
    except (ValueError, TypeError):
        return default

def detect_price_column(df):
    """Detect the actual price column by checking for numeric values."""
    possible_names = ['price', 'cost', 'fare', 'amount', 'rate', 'charges', 'Price']
    
    for name in possible_names:
        if name in df.columns:
            try:
                test_val = df[name].iloc[0]
                if safe_int_convert(test_val, None) is not None:
                    return name
            except:
                continue
    
    for col in df.columns:
        try:
            test_vals = df[col].head(3)
            numeric_count = sum(1 for v in test_vals if safe_int_convert(v, None) is not None)
            if numeric_count >= 2:
                return col
        except:
            continue
    
    return df.columns[-1] if not df.empty else None

# =======================
# MACHINE LEARNING COMPONENT
# =======================
@st.cache_data
def get_hotel_recommendation_model(df):
    if df.empty:
        return None
        
    category_col = next((col for col in ['category', 'type', 'hotel_type'] if col in df.columns), None)
    features_col = next((col for col in ['features', 'amenities', 'facilities'] if col in df.columns), None)
    rating_col = next((col for col in ['rating', 'ratings', 'star_rating'] if col in df.columns), None)
    
    combined = ""
    if category_col:
        combined += df[category_col].astype(str) + ' '
    if features_col:
        combined += df[features_col].astype(str) + ' '
    if rating_col:
        combined += df[rating_col].astype(str)
    
    df['combined_features'] = combined
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def recommend_hotels(df, selection_index, cosine_sim_matrix):
    if cosine_sim_matrix is None or df.empty:
        return pd.DataFrame()
        
    sim_scores = list(enumerate(cosine_sim_matrix[selection_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    hotel_indices = [i[0] for i in sim_scores]
    return df.iloc[hotel_indices]

# =======================
# SESSION STATE
# =======================
def initialize_session():
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
    if 'booking_data' not in st.session_state:
        st.session_state.booking_data = {
            "transport": {}, 
            "hotel": {}, 
            "sightseeing": {}, 
            "event": {}, 
            "final_package": {},
            "personal": {}
        }
    # personal_details stored separately for easier sidebar binding
    if 'personal_details' not in st.session_state:
        st.session_state.personal_details = {
            "full_name": "",
            "age": "",
            "gender": "",
            "contact": "",
            "email": "",
            "address": ""
        }

# =======================
# STEP 1: TRANSPORT BOOKING
# =======================
def step_1_transport():
    st.header("Step 1: Transport Booking ✈")
    transport_type = st.radio("Choose transport type:", ('Flight', 'Train', 'Bus', 'Taxi'), key='transport_radio')
    
    if transport_type == 'Flight':
        handle_flight_booking()
    elif transport_type == 'Train':
        handle_train_booking()
    elif transport_type == 'Bus':
        handle_bus_booking()
    elif transport_type == 'Taxi':
        handle_taxi_booking()

def handle_flight_booking():
    df = FLIGHTS_DF
    if df.empty:
        st.error("No flight data available.")
        return
        
    st.subheader("Available Flights")
    
    source_col = next((col for col in ['Source', 'source', 'from', 'origin', 'departure'] if col in df.columns), df.columns[0])
    dest_col = next((col for col in ['Destination', 'destination', 'to', 'arrival'] if col in df.columns), df.columns[1] if len(df.columns) > 1 else df.columns[0])
    airline_col = next((col for col in ['Airline', 'airline', 'carrier', 'flight_name'] if col in df.columns), df.columns[2] if len(df.columns) > 2 else df.columns[0])
    price_col = detect_price_column(df)
    
    flight_options = {}
    for idx, row in df.iterrows():
        source = get_column_value(row, [source_col], "Unknown")
        dest = get_column_value(row, [dest_col], "Unknown")
        airline = get_column_value(row, [airline_col], "Unknown")
        price = safe_int_convert(get_column_value(row, [price_col], 0))
        key = f"{source} → {dest} | {airline} | ₹{price}"
        flight_options[key] = row
    
    selected_option = st.selectbox("Select a flight:", list(flight_options.keys()), key='flight_select')
    
    if selected_option:
        flight = flight_options[selected_option]
        seat_prices = {"Window (+₹300)": 300, "Middle (+₹150)": 150, "Aisle (+₹200)": 200, "Last Seat (+₹100)": 100, "Free Seat (+₹0)": 0}
        seat_choice = st.selectbox("Seat Preference:", list(seat_prices.keys()), key='seat_select')
        addon_options = {"Lounge Access (+₹1000)": 1000, "Extra Baggage (+₹800)": 800, "Skip Queue (+₹500)": 500}
        addons = st.multiselect("Select Add-ons:", list(addon_options.keys()), key='addon_select')
        
        seat_cost = seat_prices[seat_choice]
        addon_cost = sum(addon_options[a] for a in addons)
        base_price = safe_int_convert(get_column_value(flight, [price_col], 0))
        price = base_price + seat_cost + addon_cost
        
        st.session_state.booking_data['transport'] = {
            "mode": "Flight",
            "source": get_column_value(flight, [source_col], "Unknown"),
            "destination": get_column_value(flight, [dest_col], "Unknown"),
            "price": price,
            "base_price": base_price
        }
        st.success(f"✅ Flight selected: {get_column_value(flight, [source_col])} → {get_column_value(flight, [dest_col])}. Price: ₹{price}")
        
        if st.button("Continue to Hotel Booking", key='next_step_1'):
            st.session_state.current_step = 2
            st.rerun()

def handle_train_booking():
    df = TRAIN_DF
    if df.empty:
        st.warning("No train data available.")
        return
    
    st.subheader("Available Trains")
    source_col = next((col for col in ['Source', 'source', 'from', 'origin', 'departure'] if col in df.columns), df.columns[0])
    dest_col = next((col for col in ['Destination', 'destination', 'to', 'arrival'] if col in df.columns), df.columns[1] if len(df.columns) > 1 else df.columns[0])
    train_col = next((col for col in ['Train_Name', 'train', 'train_name', 'name'] if col in df.columns), df.columns[2] if len(df.columns) > 2 else df.columns[0])
    price_col = detect_price_column(df)
    
    train_options = {}
    for idx, row in df.iterrows():
        source = get_column_value(row, [source_col], "Unknown")
        dest = get_column_value(row, [dest_col], "Unknown")
        train = get_column_value(row, [train_col], "Unknown")
        price = safe_int_convert(get_column_value(row, [price_col], 0))
        key = f"{source} → {dest} | {train} | ₹{price}"
        train_options[key] = row
    
    selected_option = st.selectbox("Select a train:", list(train_options.keys()), key='train_select')
    
    if selected_option:
        train = train_options[selected_option]
        class_prices = {"Sleeper (+₹0)": 0, "3AC (+₹500)": 500, "2AC (+₹800)": 800, "1AC (+₹1200)": 1200}
        class_choice = st.selectbox("Class Preference:", list(class_prices.keys()), key='class_select')
        addon_options = {"Meals (+₹200)": 200, "Bedding (+₹100)": 100}
        addons = st.multiselect("Select Add-ons:", list(addon_options.keys()), key='train_addon_select')
        
        class_cost = class_prices[class_choice]
        addon_cost = sum(addon_options[a] for a in addons)
        base_price = safe_int_convert(get_column_value(train, [price_col], 0))
        price = base_price + class_cost + addon_cost
        source = get_column_value(train, [source_col], "Unknown")
        destination = get_column_value(train, [dest_col], "Unknown")
        
        save_transport_data("Train", source, destination, price, base_price)
        st.success(f"✅ Train selected: {source} → {destination}. Price: ₹{price}")
        
        if st.button("Continue to Hotel Booking", key='next_step_1_train'):
            st.session_state.current_step = 2
            st.rerun()

def handle_bus_booking():
    df = BUS_DF
    if df.empty:
        st.warning("No bus data available.")
        return
    
    st.subheader("Available Buses")
    source_col = next((col for col in ['Source', 'source', 'from', 'origin', 'departure'] if col in df.columns), df.columns[0])
    dest_col = next((col for col in ['Destination', 'destination', 'to', 'arrival'] if col in df.columns), df.columns[1] if len(df.columns) > 1 else df.columns[0])
    bus_col = next((col for col in ['Operator', 'bus', 'bus_name', 'name', 'operator'] if col in df.columns), df.columns[2] if len(df.columns) > 2 else df.columns[0])
    price_col = detect_price_column(df)
    
    bus_options = {}
    for idx, row in df.iterrows():
        source = get_column_value(row, [source_col], "Unknown")
        dest = get_column_value(row, [dest_col], "Unknown")
        bus = get_column_value(row, [bus_col], "Unknown")
        price = safe_int_convert(get_column_value(row, [price_col], 0))
        key = f"{source} → {dest} | {bus} | ₹{price}"
        bus_options[key] = row
    
    selected_option = st.selectbox("Select a bus:", list(bus_options.keys()), key='bus_select')
    
    if selected_option:
        bus = bus_options[selected_option]
        seat_prices = {"Seater (+₹0)": 0, "Semi-Sleeper (+₹200)": 200, "Sleeper (+₹400)": 400}
        seat_choice = st.selectbox("Seat Type:", list(seat_prices.keys()), key='bus_seat_select')
        
        seat_cost = seat_prices[seat_choice]
        base_price = safe_int_convert(get_column_value(bus, [price_col], 0))
        price = base_price + seat_cost
        source = get_column_value(bus, [source_col], "Unknown")
        destination = get_column_value(bus, [dest_col], "Unknown")
        
        save_transport_data("Bus", source, destination, price, base_price)
        st.success(f"✅ Bus selected: {source} → {destination}. Price: ₹{price}")
        
        if st.button("Continue to Hotel Booking", key='next_step_1_bus'):
            st.session_state.current_step = 2
            st.rerun()

def handle_taxi_booking():
    df = TAXI_DF
    if df.empty:
        st.warning("No taxi data available.")
        return
    
    st.subheader("Available Taxi Services")
    source_col = next((col for col in ['Source', 'source', 'from', 'origin', 'pickup'] if col in df.columns), df.columns[0])
    dest_col = next((col for col in ['Destination', 'destination', 'to', 'drop'] if col in df.columns), df.columns[1] if len(df.columns) > 1 else df.columns[0])
    taxi_col = next((col for col in ['Travel_Company', 'taxi', 'service', 'operator', 'name'] if col in df.columns), df.columns[2] if len(df.columns) > 2 else df.columns[0])
    price_col = detect_price_column(df)
    
    taxi_options = {}
    for idx, row in df.iterrows():
        source = get_column_value(row, [source_col], "Unknown")
        dest = get_column_value(row, [dest_col], "Unknown")
        taxi = get_column_value(row, [taxi_col], "Unknown")
        price = safe_int_convert(get_column_value(row, [price_col], 0))
        key = f"{source} → {dest} | {taxi} | ₹{price}"
        taxi_options[key] = row
    
    selected_option = st.selectbox("Select a taxi:", list(taxi_options.keys()), key='taxi_select')
    
    if selected_option:
        taxi = taxi_options[selected_option]
        car_prices = {"Sedan (+₹0)": 0, "SUV (+₹500)": 500, "Luxury (+₹1500)": 1500}
        car_choice = st.selectbox("Car Type:", list(car_prices.keys()), key='car_select')
        
        car_cost = car_prices[car_choice]
        base_price = safe_int_convert(get_column_value(taxi, [price_col], 0))
        price = base_price + car_cost
        source = get_column_value(taxi, [source_col], "Unknown")
        destination = get_column_value(taxi, [dest_col], "Unknown")
        
        save_transport_data("Taxi", source, destination, price, base_price)
        st.success(f"✅ Taxi selected: {source} → {destination}. Price: ₹{price}")
        
        if st.button("Continue to Hotel Booking", key='next_step_1_taxi'):
            st.session_state.current_step = 2
            st.rerun()

def save_transport_data(mode, source, destination, price, base_price=None):
    st.session_state.booking_data['transport'] = {
        "mode": mode,
        "source": source,
        "destination": destination,
        "price": price,
        "base_price": base_price if base_price else price
    }

# =======================
# STEP 2: HOTEL BOOKING
# =======================
def step_2_hotel():
    destination_city = st.session_state.booking_data['transport'].get('destination')
    st.header(f"Step 2: Hotel Booking in {destination_city} 🏨")
    
    if not destination_city:
        st.error("Please complete Transport Booking first.")
        if st.button("Go back to Step 1", key='back_step_2'):
            st.session_state.current_step = 1
            st.rerun()
        return

    city_col = next((col for col in ['city', 'location', 'place', 'destination'] if col in HOTELS_DF.columns), HOTELS_DF.columns[0] if not HOTELS_DF.empty else None)
    
    if city_col:
        city_hotels = HOTELS_DF[HOTELS_DF[city_col] == destination_city].reset_index(drop=True)
    else:
        city_hotels = HOTELS_DF.reset_index(drop=True)
    
    if city_hotels.empty:
        st.warning(f"No hotels found in {destination_city}. Skipping hotel booking.")
        st.session_state.booking_data['hotel'] = {"hotel": None, "price": 0}
        if st.button("Continue to Sightseeing", key='skip_step_2'):
            st.session_state.current_step = 3
            st.rerun()
        return

    hotel_col = next((col for col in ['name', 'hotel', 'hotel_name'] if col in city_hotels.columns), city_hotels.columns[0])
    category_col = next((col for col in ['category', 'type', 'hotel_type'] if col in city_hotels.columns), None)
    rating_col = next((col for col in ['rating', 'ratings', 'star_rating'] if col in city_hotels.columns), None)
    price_col = detect_price_column(city_hotels)

    st.subheader("Hotel Recommendation System (ML-Powered)")
    preference = st.radio("What type of hotel are you looking for?", 
                          ('All', 'High-Rated (4.5+)', 'Luxury Category', 'Budget Category'), 
                          key='hotel_pref')
    
    display_hotels = city_hotels.copy()
    if preference == 'High-Rated (4.5+)' and rating_col:
        display_hotels = city_hotels[city_hotels[rating_col] >= 4.5]
    elif preference == 'Luxury Category' and category_col:
        display_hotels = city_hotels[city_hotels[category_col].str.contains('Luxury|Grand|Royal', case=False, na=False)]
    elif preference == 'Budget Category' and category_col:
        display_hotels = city_hotels[city_hotels[category_col].str.contains('Budget|Economy|Comfort', case=False, na=False)]

    st.markdown("---")
    st.subheader("Available Hotels (Filtered)")
    
    hotel_options = {}
    for idx, row in display_hotels.iterrows():
        hotel_name = get_column_value(row, [hotel_col], "Unknown Hotel")
        category = get_column_value(row, [category_col], "Standard") if category_col else "Standard"
        rating = get_column_value(row, [rating_col], "N/A") if rating_col else "N/A"
        price = safe_int_convert(get_column_value(row, [price_col], 0))
        key = f"{hotel_name} ({category}) — Rating {rating}★ — ₹{price}"
        hotel_options[key] = idx
    
    selected_option_key = st.selectbox("Select your hotel:", list(hotel_options.keys()), key='hotel_select')
    
    if selected_option_key:
        selected_index = hotel_options[selected_option_key]
        chosen = display_hotels.loc[selected_index]
        
        col1, col2 = st.columns(2)
        with col1:
            check_in = st.date_input("Check-In Date:", min_value=date.today(), key='check_in')
        with col2:
            check_out = st.date_input("Check-Out Date:", min_value=check_in, key='check_out')
        
        nights = get_nights(check_in, check_out)
        st.write(f"Nights: {nights}")
        
        if nights < 1:
            st.warning("Check-Out must be after Check-In.")
            return

        addon_options = {"Swimming Pool (+₹1000)": 1000, "Bar (+₹1500)": 1500, "Gym (+₹1500)": 1500, "Spa (+₹2000)": 2000}
        addons = st.multiselect("Select Add-ons:", list(addon_options.keys()), key='hotel_addon_select')
        addon_cost = sum(addon_options[a] for a in addons)
        base_price = safe_int_convert(get_column_value(chosen, [price_col], 0)) * nights
        total_price = base_price + addon_cost
        st.success(f"Hotel cost for {nights} nights: ₹{base_price}. Total with Add-ons: ₹{total_price}")

        st.markdown("---")
        st.subheader("You might also like (ML Recommendation)")
        try:
            full_index = city_hotels[city_hotels[hotel_col] == chosen[hotel_col]].index[0]
            cosine_sim = get_hotel_recommendation_model(city_hotels)
            if cosine_sim is not None:
                recommended_hotels = recommend_hotels(city_hotels, full_index, cosine_sim)
                
                if not recommended_hotels.empty:
                    display_cols = [hotel_col]
                    if category_col:
                        display_cols.append(category_col)
                    if rating_col:
                        display_cols.append(rating_col)
                    display_cols.append(price_col)
                    
                    st.table(recommended_hotels[display_cols].rename(columns={
                        hotel_col: 'Name',
                        category_col: 'Type' if category_col else 'Category',
                        rating_col: 'Rating' if rating_col else 'Stars',
                        price_col: 'Price/Night'
                    }))
                else:
                    st.info("No similar hotels found for recommendation.")
            else:
                st.info("Recommendation system not available.")
        except Exception as e:
            st.info(f"Could not generate recommendations: {e}")
        
        if st.button("Finalize Hotel & Continue", key='next_step_2'):
            st.session_state.booking_data['hotel'] = {
                "hotel": get_column_value(chosen, [hotel_col], "Unknown Hotel"), 
                "price": total_price, 
                "nights": nights,
                "base_price": base_price
            }
            st.session_state.current_step = 3
            st.rerun()

# =======================
# STEP 3: SIGHTSEEING
# =======================
def step_3_sightseeing():
    destination_city = st.session_state.booking_data['transport'].get('destination')
    st.header(f"Step 3: Sightseeing Booking in {destination_city} 📸")
    
    if not destination_city:
        st.error("Please complete Transport Booking first.")
        if st.button("Go back to Step 1", key='back_step_3'):
            st.session_state.current_step = 1
            st.rerun()
        return

    city_col = next((col for col in ['city', 'location', 'place', 'destination'] if col in SIGHTS_DF.columns), SIGHTS_DF.columns[0] if not SIGHTS_DF.empty else None)
    place_col = next((col for col in ['place', 'name', 'attraction', 'sight'] if col in SIGHTS_DF.columns), SIGHTS_DF.columns[1] if len(SIGHTS_DF.columns) > 1 else SIGHTS_DF.columns[0])
    price_col = detect_price_column(SIGHTS_DF) if not SIGHTS_DF.empty else None

    if city_col:
        city_sights = SIGHTS_DF[SIGHTS_DF[city_col] == destination_city]
    else:
        city_sights = SIGHTS_DF
    
    if city_sights.empty:
        st.warning(f"No sightseeing options found in {destination_city}. Skipping.")
        st.session_state.booking_data['sightseeing'] = {"place": None, "price": 0}
        if st.button("Continue to Events", key='skip_step_3'):
            st.session_state.current_step = 4
            st.rerun()
        return

    sight_options = {}
    for idx, row in city_sights.iterrows():
        place = get_column_value(row, [place_col], "Unknown Place")
        price = safe_int_convert(get_column_value(row, [price_col], 0))
        key = f"{place} (₹{price} per ticket)"
        sight_options[key] = row
    
    selected_sights = st.multiselect("Select Sightseeing Places:", list(sight_options.keys()), key='sight_select')
    
    if selected_sights:
        tickets = st.number_input("Number of Tickets:", min_value=1, value=1, step=1, key='sight_tickets')
        total_price = 0
        booked_places = []
        for sight_key in selected_sights:
            sight = sight_options[sight_key]
            total_price += safe_int_convert(get_column_value(sight, [price_col], 0)) * tickets
            booked_places.append(get_column_value(sight, [place_col], "Unknown"))
        
        st.success(f"Sightseeing total for {tickets} ticket(s): ₹{total_price}")
        
        if st.button("Continue to Events", key='next_step_3'):
            st.session_state.booking_data['sightseeing'] = {
                "place": ", ".join(booked_places),
                "price": total_price
            }
            st.session_state.current_step = 4
            st.rerun()
    else:
        st.session_state.booking_data['sightseeing'] = {"place": None, "price": 0}
        if st.button("Skip Sightseeing & Continue to Events", key='skip_step_3_2'):
            st.session_state.current_step = 4
            st.rerun()

# =======================
# STEP 4: EVENTS
# =======================
def step_4_events():
    destination_city = st.session_state.booking_data['transport'].get('destination')
    st.header(f"Step 4: Events Booking in {destination_city} 🎤")
    
    if EVENTS_DF.empty:
        st.warning("No event data available.")
        st.session_state.booking_data['event'] = {"event_name": None, "price": 0}
        if st.button("Continue to Package Comparison", key='skip_events'):
            st.session_state.current_step = 5
            st.rerun()
        return
    
    city_col = next((col for col in ['city', 'location', 'place'] if col in EVENTS_DF.columns), EVENTS_DF.columns[0])
    event_col = next((col for col in ['event_name', 'name', 'event', 'title'] if col in EVENTS_DF.columns), EVENTS_DF.columns[1] if len(EVENTS_DF.columns) > 1 else EVENTS_DF.columns[0])
    price_col = detect_price_column(EVENTS_DF)
    
    # Filter events by destination city
    if city_col:
        city_events = EVENTS_DF[EVENTS_DF[city_col] == destination_city]
    else:
        city_events = EVENTS_DF
    
    if city_events.empty:
        st.warning(f"No events found in {destination_city}.")
        city_events = EVENTS_DF  # Show all events if none in destination city
    
    event_options = {}
    for idx, row in city_events.iterrows():
        event_name = get_column_value(row, [event_col], "Unknown Event")
        event_city = get_column_value(row, [city_col], "Unknown City")
        price = safe_int_convert(get_column_value(row, [price_col], 0))
        key = f"{event_name} in {event_city} (₹{price})"
        event_options[key] = row
    
    selected_option = st.selectbox("Choose an Event (Optional):", ['None'] + list(event_options.keys()), key='event_select')
    event_price = 0
    event_name = None
    
    if selected_option != 'None':
        event = event_options[selected_option]
        event_price = safe_int_convert(get_column_value(event, [price_col], 0))
        event_name = get_column_value(event, [event_col], "Unknown Event")
        st.success(f"Event selected: {event_name}. Price: ₹{event_price}")
    
    if st.button("Continue to Package Comparison", key='next_step_4'):
        st.session_state.booking_data['event'] = {"event_name": event_name, "price": event_price}
        st.session_state.current_step = 5
        st.rerun()

# =======================
# STEP 5: PACKAGE COMPARISON
# =======================
def step_5_comparison():
    st.header("Step 5: Package Comparison & Checkout 💰")
    
    t_price = st.session_state.booking_data['transport'].get('price', 0)
    h_price = st.session_state.booking_data['hotel'].get('price', 0)
    s_price = st.session_state.booking_data['sightseeing'].get('price', 0)
    e_price = st.session_state.booking_data['event'].get('price', 0)
    
    if t_price == 0:
        st.error("Please complete Transport Booking (Step 1).")
        if st.button("Go back to Step 1", key='back_step_5'):
            st.session_state.current_step = 1
            st.rerun()
        return

    st.subheader("Your Base Itinerary Costs:")
    st.info(f"Transport: ₹{t_price} | Hotel: ₹{h_price} | Sightseeing: ₹{s_price} | Event: ₹{e_price}")
    
    agencies = [
        {"name": "SkyHigh Travels", "offer": ("percent", 0.10)},
        {"name": "DreamLine Holidays", "offer": ("flat", 1500)},
        {"name": "WanderWorld Tours", "offer": ("percent", 0.08)},
        {"name": "EcoEscape Ventures", "offer": ("flat", 1000)},
        {"name": "RoyalRoute Travels", "offer": ("percent", 0.12)},
    ]
    
    packages = []

    with st.spinner('Preparing travel packages from multiple agencies...'):
        time.sleep(1)
        for agency in agencies:
            transport_price = int(t_price * random.uniform(0.9, 1.1))
            hotel_price = int(h_price * random.uniform(0.9, 1.1))
            sightseeing_price = int(s_price * random.uniform(0.9, 1.1))
            event_price = int(e_price * random.uniform(0.9, 1.1))
            
            subtotal = transport_price + hotel_price + sightseeing_price + event_price
            
            if agency["offer"][0] == "percent":
                discount = int(subtotal * agency["offer"][1])
                offer_str = f"{int(agency['offer'][1]*100)}% off"
            else:
                discount = agency["offer"][1]
                offer_str = f"Flat ₹{discount}"
            
            discount = min(discount, subtotal)
            subtotal_after_discount = subtotal - discount
            gst = calculate_gst(subtotal_after_discount)
            total = subtotal_after_discount + gst
            
            packages.append({
                "Agency": agency["name"],
                "Transport": transport_price,
                "Hotel": hotel_price,
                "Sightseeing": sightseeing_price,
                "Event": event_price,
                "Offer": offer_str,
                "Discount": discount,
                "GST": gst,
                "Total": total
            })

    packages_df = pd.DataFrame(packages).sort_values(by='Total')
    st.subheader("📊 Final Package Comparison")
    st.dataframe(
        packages_df.style.highlight_min(subset=['Total'], axis=0, color='lightgreen'),
        use_container_width=True
    )
    
    agency_list = packages_df['Agency'].tolist()
    selected_agency_name = st.selectbox("👉 Choose the best package:", agency_list, key='final_agency_select')
    
    if st.button("Confirm Booking and Checkout", key='checkout_button'):
        selected_pkg = packages_df[packages_df['Agency'] == selected_agency_name].iloc[0].to_dict()
        st.session_state.booking_data['final_package'] = selected_pkg
        st.session_state.current_step = 6
        st.rerun()

# =======================
# STEP 6: RECEIPT
# =======================
def step_6_receipt():
    st.header("Step 6: Booking Confirmation 🎉")
    pkg = st.session_state.booking_data.get('final_package', {})
    
    if not pkg:
        st.error("No package was finalized.")
        if st.button("Go back to Step 5", key='back_step_6'):
            st.session_state.current_step = 5
            st.rerun()
        return

    st.subheader("✅ FINAL BOOKING RECEIPT")
    st.markdown("---")
    
    # Display personal details on receipt (if present)
    pdets = st.session_state.personal_details if 'personal_details' in st.session_state else {}
    if pdets:
        st.markdown("### Passenger / Booker Details")
        st.write(f"**Name:** {pdets.get('full_name','')}")
        st.write(f"**Age:** {pdets.get('age','')}")
        st.write(f"**Gender:** {pdets.get('gender','')}")
        st.write(f"**Contact:** {pdets.get('contact','')}")
        st.write(f"**Email:** {pdets.get('email','')}")
        st.write(f"**Address:** {pdets.get('address','')}")
        st.markdown("---")
    
    col1, col2 = st.columns(2)
    col1.metric("Agency", pkg['Agency'])
    col2.metric("Total Payable", f"₹{pkg['Total']}")
    
    st.markdown("### Cost Breakdown")
    col3, col4, col5, col6 = st.columns(4)
    col3.info(f"Transport: ₹{pkg['Transport']}")
    col4.info(f"Hotel: ₹{pkg['Hotel']}")
    col5.info(f"Sightseeing: ₹{pkg['Sightseeing']}")
    col6.info(f"Event: ₹{pkg['Event']}")

    st.markdown("### Summary")
    st.write(f"Offer Applied: {pkg['Offer']} (Discount: ₹{pkg['Discount']})")
    st.write(f"GST ({GST_RATE*100}%): ₹{pkg['GST']}")
    
    # Store the personal details inside booking_data (so they are part of final data structure)
    st.session_state.booking_data['personal'] = st.session_state.personal_details.copy()
    
    st.balloons()
    st.success("✨ Thank you for choosing our ML-Powered Travel System! ✨")

# =======================
# MAIN APP
# =======================
def main():
    st.set_page_config(page_title="ML-Travel Planner", layout="wide")
    initialize_session()
    st.title("Intelligent Travel Itinerary Planner 🌍✈")
    
    # -----------------------
    # Personal Details in Sidebar (non-invasive: can be edited anytime)
    # -----------------------
    st.sidebar.title("Your Details")
    # Pre-fill with existing session values
    pd_fullname = st.sidebar.text_input("Full Name", value=st.session_state.personal_details.get('full_name',''))
    pd_age = st.sidebar.text_input("Age", value=st.session_state.personal_details.get('age',''))
    pd_gender = st.sidebar.selectbox("Gender", options=["", "Male", "Female", "Other"], index=0 if st.session_state.personal_details.get('gender','')=='' else (["", "Male", "Female", "Other"].index(st.session_state.personal_details.get('gender',''))), key='pd_gender_select')
    pd_contact = st.sidebar.text_input("Contact Number", value=st.session_state.personal_details.get('contact',''))
    pd_email = st.sidebar.text_input("Email ID", value=st.session_state.personal_details.get('email',''))
    pd_address = st.sidebar.text_area("Address (optional)", value=st.session_state.personal_details.get('address',''), height=100)
    
    # Save immediately into session_state and booking_data['personal']
    st.session_state.personal_details['full_name'] = pd_fullname
    st.session_state.personal_details['age'] = pd_age
    st.session_state.personal_details['gender'] = st.session_state.get('pd_gender_select', st.session_state.personal_details.get('gender',''))
    st.session_state.personal_details['contact'] = pd_contact
    st.session_state.personal_details['email'] = pd_email
    st.session_state.personal_details['address'] = pd_address
    # Also ensure booking_data has a copy
    st.session_state.booking_data['personal'] = st.session_state.personal_details.copy()
    
    st.sidebar.markdown("---")
    # Quick view of transport data for convenience
    st.sidebar.markdown("**Selected Transport**")
    st.sidebar.json(st.session_state.booking_data.get('transport', {}))
    st.sidebar.markdown("---")
    st.sidebar.markdown("Navigate through the steps below:")
    
    st.sidebar.title("Navigation")
    steps = {
        1: "Transport",
        2: "Hotel (ML-Enabled)",
        3: "Sightseeing",
        4: "Events",
        5: "Compare & Checkout",
        6: "Receipt"
    }
    
    for step_num, step_name in steps.items():
        if step_num < st.session_state.current_step:
            if st.sidebar.button(f"Go to {step_name}", key=f"nav_{step_num}"):
                st.session_state.current_step = step_num
                st.rerun()
        elif step_num == st.session_state.current_step:
            st.sidebar.markdown(f"*Current: {step_name}*")
        else:
            st.sidebar.markdown(f"{step_name}")
    
    st.sidebar.markdown("---")
    st.sidebar.json(st.session_state.booking_data['transport'])

    if st.session_state.current_step == 1:
        step_1_transport()
    elif st.session_state.current_step == 2:
        step_2_hotel()
    elif st.session_state.current_step == 3:
        step_3_sightseeing()
    elif st.session_state.current_step == 4:
        step_4_events()
    elif st.session_state.current_step == 5:
        step_5_comparison()
    elif st.session_state.current_step == 6:
        step_6_receipt()

if __name__ == "__main__":
    main()