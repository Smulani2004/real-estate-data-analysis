import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Real Estate Price Prediction", page_icon="🏠", layout="wide")

st.markdown("""
<style>
.main { padding: 1rem 2rem; }
.stMetric { background: linear-gradient(135deg, #667eea, #764ba2); border-radius: 10px; padding: 15px; color: white; }
.insight-card { background: #f8f9fa; border-left: 4px solid #667eea; padding: 15px; border-radius: 0 10px 10px 0; }
h1 { color: #2d3748; font-weight: 700; }
.stButton>button { background: linear-gradient(135deg, #667eea, #764ba2); color: white; border-radius: 25px; padding: 12px 30px; font-weight: 600; border: none; }
.stSlider>div>div>div>div { background: linear-gradient(90deg, #667eea, #764ba2); border-radius: 10px; height: 8px; }
.section-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 8px 16px; border-radius: 8px; font-size: 13px; font-weight: 600; margin-bottom: 15px; display: inline-block; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            return pickle.load(f)
    except:
        return None

@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("cleaned_housing_data.csv")
    except:
        return None

def calc_price_per_sqft(area, df):
    avg = df['price_per_sqft'].mean()
    return round(avg * (1.2 if area < 3000 else 1.0 if area < 6000 else 0.8), 2)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/619/619032.png", width=80)
    st.title("🏠 Real Estate AI")
    st.markdown("---")
    st.info("AI-powered property valuation using Linear Regression.")
    show_dataset = st.toggle("📋 Show Dataset", value=False)

st.title("🏠 Real Estate Price Prediction System")
st.markdown("### AI-Powered Property Valuation Dashboard")
st.markdown("---")

model, df = load_model(), load_dataset()
if model is None or df is None:
    st.error("⚠️ Failed to load resources. Check model.pkl and cleaned_housing_data.csv.")
    st.stop()

for key, val in [("a", 3000), ("b", 3), ("ba", 1), ("s", 2), ("p", 1)]:
    if key not in st.session_state:
        st.session_state[key] = val

st.markdown("### 🏡 Property Details")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("<div class='section-header'>📐 Property Size</div>", unsafe_allow_html=True)
    min_a, max_a = int(df['area'].min()), int(df['area'].max())
    area = st.slider(f"Area — {st.session_state.a:,} sq ft", min_a, max_a, st.session_state.a, 100, key="sa")
    st.session_state.a = area

with c2:
    st.markdown("<div class='section-header'>🏗️ Structure</div>", unsafe_allow_html=True)
    min_b, max_b = int(df['bedrooms'].min()), int(df['bedrooms'].max())
    bedrooms = st.slider(f"Bedrooms — {st.session_state.b}", min_b, max_b, st.session_state.b, 1, key="sb")
    st.session_state.b = bedrooms
    
    bath_min, bath_max = int(df['bathrooms'].min()), int(df['bathrooms'].max())
    if bath_min == bath_max:
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=bath_min, step=1)
    else:
        bathrooms = st.slider(f"Bathrooms — {st.session_state.ba}", bath_min, bath_max, st.session_state.ba, 1, key="sba")
        st.session_state.ba = bathrooms
    
    min_s, max_s = int(df['stories'].min()), int(df['stories'].max())
    stories = st.slider(f"Stories — {st.session_state.s}", min_s, max_s, st.session_state.s, 1, key="ss")
    st.session_state.s = stories

with c3:
    st.markdown("<div class='section-header'>✨ Amenities</div>", unsafe_allow_html=True)
    min_p, max_p = int(df['parking'].min()), int(df['parking'].max())
    parking = st.slider(f"Parking — {st.session_state.p}", min_p, max_p, st.session_state.p, 1, key="sp")
    st.session_state.p = parking
    
    ca, cb = st.columns(2)
    with ca:
        mainroad = st.toggle("🛣️ Main Road", True)
        guestroom = st.toggle("🛏️ Guest Room")
        basement = st.toggle("🏚️ Basement")
    with cb:
        hotwater = st.toggle("🔥 Hot Water")
        ac = st.toggle("❄️ AC")
        prefarea = st.toggle("⭐ Pref. Area")

st.markdown("---")
st.markdown("<div class='section-header'>🛋️ Furnishing Status</div>", unsafe_allow_html=True)
furnish_opts = {"Unfurnished": 0, "Semi-Furnished": 1, "Fully Furnished": 2}
furnish = st.selectbox("Select furnishing level", list(furnish_opts.keys()), 1)

if st.button("🔮 Predict Price", use_container_width=True):
    with st.spinner("🤖 Analyzing..."):
        price_per_sqft = calc_price_per_sqft(area, df)
        total_rooms = bedrooms + bathrooms
        vals = [int(x) for x in [mainroad, guestroom, basement, hotwater, ac, prefarea, furnish_opts[furnish]]]
        
        inp = pd.DataFrame({
            "area": [area], "bedrooms": [bedrooms], "bathrooms": [bathrooms], "stories": [stories],
            "mainroad": [vals[0]], "guestroom": [vals[1]], "basement": [vals[2]],
            "hotwaterheating": [vals[3]], "airconditioning": [vals[4]], "parking": [parking],
            "prefarea": [vals[5]], "furnishingstatus": [vals[6]],
            "price_per_sqft": [price_per_sqft], "total_rooms": [total_rooms]
        })
        
        pred = model.predict(inp)[0]
        avg_price = df['price'].mean()
        med_price = df['price'].median()
        cat = "🔥 Premium" if pred > avg_price * 1.2 else "📈 Above Average" if pred > avg_price else "📉 Below Average"
        
        st.success("✅ Prediction complete!")
        
        st.markdown("### 📊 Valuation Summary")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("💰 Estimated Price", f"₹{pred:,.0f}", f"{((pred - avg_price) / avg_price * 100):+.1f}% vs Avg")
        k2.metric("📏 Price/Sq Ft", f"₹{price_per_sqft:,.0f}")
        k3.metric("🚪 Total Rooms", str(total_rooms), f"{bedrooms} Bed + {bathrooms} Bath")
        k4.metric("🏷️ Category", cat, f"vs ₹{avg_price:,.0f} avg")
        
        st.markdown("---")
        st.markdown("### 📈 Market Analysis")
        i1, i2 = st.columns([2, 1])
        
        with i1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta", value=pred,
                number={"prefix": "₹", "font": {"size": 24}},
                delta={"reference": avg_price, "prefix": "vs Avg: "},
                title={"text": "Price vs Market Average"},
                gauge={"axis": {"range": [0, max(df["price"].max(), pred) * 1.1]}, "bar": {"color": "darkblue"},
                       "steps": [{"range": [0, avg_price * 0.8], "color": "lightcoral"},
                                {"range": [avg_price * 0.8, avg_price * 1.2], "color": "lightyellow"},
                                {"range": [avg_price * 1.2, df["price"].max()], "color": "lightgreen"}],
                       "threshold": {"line": {"color": "red", "width": 4}, "value": avg_price}}
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
        
        with i2:
            st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
            st.markdown("#### 🎯 Insights")
            st.markdown(f"**{cat}**  \nThis property is {('significantly above' if pred > avg_price * 1.2 else 'above' if pred > avg_price else 'below')} market average.")
            st.markdown(f"- Avg: ₹{avg_price:,.0f}\n- Median: ₹{med_price:,.0f}\n- Diff: ₹{pred - avg_price:+,.0f}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            report = f"""REAL ESTATE PRICE PREDICTION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PROPERTY: Area {area} sq ft | {bedrooms} Bed | {bathrooms} Bath | {stories} Stories | {parking} Parking
Amenities: Main Road {'Yes' if mainroad else 'No'}, Guest Room {'Yes' if guestroom else 'No'}, Basement {'Yes' if basement else 'No'}, AC {'Yes' if ac else 'No'}
Furnishing: {furnish}

PREDICTION:
- Price: ₹{pred:,.2f}
- Category: {cat}
- vs Market Avg: {((pred - avg_price) / avg_price * 100):+.2f}%
"""
            st.download_button("📥 Download Report", report, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        st.markdown("---")
        st.markdown("### 📊 Visualizations")
        v1, v2 = st.columns(2)
        
        with v1:
            st.markdown("#### Price vs Area")
            fig_scatter = px.scatter(df, x="area", y="price", opacity=0.6, color_discrete_sequence=["#667eea"])
            fig_scatter.add_trace(go.Scatter(x=[area], y=[pred], mode="markers", marker=dict(size=15, color="red", symbol="star"), name="Your Property"))
            fig_scatter.update_layout(height=350, margin=dict(l=40, r=20, t=30, b=40))
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with v2:
            st.markdown("#### Feature Importance")
            if hasattr(model, "coef_"):
                features = ["Area", "Bedrooms", "Bathrooms", "Stories", "Main Road", "Guest Room", "Basement", "Hot Water", "AC", "Parking", "Preferred Area", "Furnishing", "Price/SqFt", "Total Rooms"]
                imp_df = pd.DataFrame({"Feature": features, "Importance": np.abs(model.coef_)}).sort_values("Importance", ascending=True)
                fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="viridis")
                fig_imp.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=40), coloraxis_showscale=False)
                st.plotly_chart(fig_imp, use_container_width=True)

if show_dataset:
    st.markdown("---")
    st.markdown("### 📋 Dataset Preview")
    t1, t2 = st.tabs(["Sample Data", "Statistics"])
    with t1: st.dataframe(df.head(20), use_container_width=True)
    with t2: st.dataframe(df.describe(), use_container_width=True)
    
    fig_dist = px.histogram(df, x="price", nbins=30, color_discrete_sequence=["#667eea"])
    fig_dist.update_layout(height=300, margin=dict(l=40, r=20, t=30, b=40))
    st.plotly_chart(fig_dist, use_container_width=True)

st.markdown("---")
st.markdown("<div style='text-align:center;color:#718096;padding:20px;'>🏠 Real Estate Price Prediction | Built with Streamlit • Linear Regression</div>", unsafe_allow_html=True)
