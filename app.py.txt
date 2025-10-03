import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import tempfile

# =========================
# Geometry Calculation
# =========================
def annular_sector_geometry(L, B, theta_deg):
    theta = np.radians(theta_deg)
    if theta == 0:
        theta = 1e-6
    r = max(0, L/theta - B/2)
    R = r + B
    area = 0.5 * theta * (R**2 - r**2)
    return {"inner_radius": r, "outer_radius": R, "theta_rad": theta,
            "theta_deg": theta_deg, "area": area, "arc_outer": R*theta, "arc_inner": r*theta}

# =========================
# Random Ratios Generator
# =========================
def random_ratios(n):
    vals = np.random.rand(n)
    ratios = vals / np.sum(vals)
    return [round(r,3) for r in ratios]

# =========================
# Plot Sector + Rooms
# =========================
def plot_annular(L, B, theta, n_rooms, mode, ratios, scale):
    g = annular_sector_geometry(L,B,theta)
    r,R,theta_rad = g["inner_radius"], g["outer_radius"], g["theta_rad"]
    total_area = g["area"]
    
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_aspect('equal')
    ax.add_patch(patches.Wedge((0,0), R*scale, 0, theta, width=(R-r)*scale, alpha=0.2, color='skyblue'))

    rooms=[]
    colors=np.random.rand(n_rooms,3)
    
    if mode in ["radial","custom"]:
        start_angle=0
        for i,p in enumerate(ratios):
            dtheta=p*theta_rad
            ax.add_patch(patches.Wedge((0,0), R*scale, np.degrees(start_angle), np.degrees(start_angle+dtheta),
                                       width=(R-r)*scale, alpha=0.5, color=colors[i]))
            area=0.5*(R**2 - r**2)*dtheta
            rooms.append({"Room":i+1,"Area (m²)": round(area,2),"Percentage": round(area/total_area*100,2)})
            angle_mid=start_angle+dtheta/2
            ax.text((R+r)/2*scale*np.cos(angle_mid), (R+r)/2*scale*np.sin(angle_mid),
                    f'{i+1}\n{round(area,2)} m²\n{round(area/total_area*100,1)}%', ha='center', va='center', fontsize=10)
            start_angle+=dtheta
    elif mode=="tangential":
        dr=(R-r)/n_rooms
        for i in range(n_rooms):
            r1=r+i*dr
            r2=r+(i+1)*dr
            ax.add_patch(patches.Wedge((0,0), r2*scale, 0, theta, width=(r2-r1)*scale, alpha=0.5, color=colors[i]))
            area=0.5*theta*(r2**2 - r1**2)
            rooms.append({"Room":i+1,"Area (m²)": round(area,2),"Percentage": round(area/total_area*100,2)})
            ax.text((r1+r2)/2*scale,0,f'{i+1}\n{round(area,2)} m²',ha='center',va='center',fontsize=10)
    
    ax.set_title(f"Annular Sector - {mode.capitalize()} Rooms", fontsize=14)
    ax.grid(True)
    plt.tight_layout()
    return fig, pd.DataFrame(rooms)

# =========================
# Paper-ready plot
# =========================
def paper_ready_plot(L,B,theta,n_rooms,mode,ratios,scale):
    g = annular_sector_geometry(L,B,theta)
    r,R,theta_rad = g["inner_radius"], g["outer_radius"], g["theta_rad"]
    
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_aspect('equal')
    
    theta_vals=np.linspace(0,theta,100)
    x_outer=[R*scale*np.cos(t) for t in theta_vals]
    y_outer=[R*scale*np.sin(t) for t in theta_vals]
    x_inner=[r*scale*np.cos(t) for t in theta_vals]
    y_inner=[r*scale*np.sin(t) for t in theta_vals]
    ax.plot(x_outer,y_outer,'k',lw=2)
    ax.plot(x_inner,y_inner,'k',lw=2)
    ax.plot([x_inner[0],x_outer[0]],[y_inner[0],y_outer[0]],'k',lw=2)
    ax.plot([x_inner[-1],x_outer[-1]],[y_inner[-1],y_outer[-1]],'k',lw=2)

    if mode in ["radial","custom"]:
        start_angle=0
        for p in ratios[:-1]:
            dtheta=p*theta_rad
            ax.plot([r*scale*np.cos(start_angle), R*scale*np.cos(start_angle)],
                    [r*scale*np.sin(start_angle), R*scale*np.sin(start_angle)], 'r--', lw=1)
            start_angle+=dtheta
    elif mode=="tangential":
        dr=(R-r)/n_rooms
        for i in range(1,n_rooms):
            ri=r+i*dr
            theta_deg=np.linspace(0,theta,100)
            x=[ri*scale*np.cos(t) for t in theta_deg]
            y=[ri*scale*np.sin(t) for t in theta_deg]
            ax.plot(x,y,'r--',lw=1)
    
    ax.set_xticks(range(-int(R*scale)-1,int(R*scale)+2,1))
    ax.set_yticks(range(-int(R*scale)-1,int(R*scale)+2,1))
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title("Paper-ready Scaled Plot", fontsize=14)
    plt.tight_layout()
    return fig

# =========================
# Save PDF
# =========================
def save_pdf(fig1, fig2):
    tmp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    pdf = PdfPages(tmp_file.name)
    pdf.savefig(fig1)
    pdf.savefig(fig2)
    pdf.close()
    return tmp_file.name

# =========================
# Streamlit App
# =========================
st.title("📐 Annular Sector Building Design Tool (Mobile-Friendly)")

# --- Inputs ---
L = st.slider("Length L (m)", 0.0, 100.0, 20.0, 0.1)
B = st.slider("Breadth B (m)", 0.0, 30.0, 6.0, 0.1)
theta = st.slider("Angle θ (deg)", 0, 180, 90, 1)
n_rooms = st.slider("Number of Rooms", 1, 10, 3, 1)
mode = st.selectbox("Room Division Mode", ["radial","tangential","custom"])
scale = st.slider("Scale Factor (1 unit = X m)", 0.1, 10.0, 1.0, 0.1)
ratios_text = st.text_input("Custom Room Ratios (sum=1, comma-separated)", "")
randomize = st.button("🎲 Randomize Ratios")
reset = st.button("🔄 Reset Ratios to Equal")

# --- Handle session state ---
if 'ratios' not in st.session_state:
    st.session_state['ratios'] = [1/n_rooms]*n_rooms

if randomize:
    st.session_state['ratios'] = random_ratios(n_rooms)
if reset:
    st.session_state['ratios'] = [round(1/n_rooms,3)]*n_rooms
if ratios_text.strip()!="":
    try:
        vals=[float(x) for x in ratios_text.split(",")]
        if len(vals)==n_rooms and abs(sum(vals)-1)<1e-6:
            st.session_state['ratios'] = vals
    except:
        pass

ratios = st.session_state['ratios']
st.write("Current Room Ratios:", ratios)

# --- Plots ---
fig1, df_rooms = plot_annular(L,B,theta,n_rooms,mode,ratios,scale)
fig2 = paper_ready_plot(L,B,theta,n_rooms,mode,ratios,scale)

st.pyplot(fig1)
st.pyplot(fig2)

# --- Data ---
st.dataframe(df_rooms)

# --- Downloads ---
pdf_file = save_pdf(fig1, fig2)
st.download_button("📄 Download Both Plots PDF", pdf_file, file_name="annular_sector_plots.pdf")

csv_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
df_rooms.to_csv(csv_file.name, index=False)
st.download_button("📊 Download Room Areas CSV", csv_file.name, file_name="room_areas.csv")

png_file1 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
fig1.savefig(png_file1.name, dpi=300, bbox_inches='tight')
st.download_button("🖼️ Download Sector Plot PNG", png_file1.name, file_name="sector_plot.png")

png_file2 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
fig2.savefig(png_file2.name, dpi=300, bbox_inches='tight')
st.download_button("🖼️ Download Paper-ready Plot PNG", png_file2.name, file_name="paper_ready_plot.png")