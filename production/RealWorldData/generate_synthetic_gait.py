import os
import json
import numpy as np
import pandas as pd

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
PROFILE_FILE=os.path.join(BASE_DIR,"biomechanical_profiles_clean.json")

OUTPUT_DIR=os.path.join(BASE_DIR,"SyntheticUsers")

# --------------------------------------------------
# Analyze real data to extract sensor statistics
# --------------------------------------------------

def analyze_real_data():

    rows=[]
    means={"ax":[],"ay":[],"az":[]}

    for i in range(1,11):

        folder=os.path.join(BASE_DIR,f"Person{i}")

        if not os.path.exists(folder):
            continue

        for f in os.listdir(folder):

            if f.endswith(".csv"):

                df=pd.read_csv(os.path.join(folder,f))

                rows.append(len(df))

                means["ax"].append(df["ax"].mean())
                means["ay"].append(df["ay"].mean())
                means["az"].append(df["az"].mean())

    avg_rows=int(np.mean(rows))

    gravity_axis=max(means,key=lambda k:abs(np.mean(means[k])))

    gravity_value=np.mean(means[gravity_axis])

    return avg_rows,gravity_axis,gravity_value


# --------------------------------------------------
# Derive biomechanical gait parameters
# --------------------------------------------------

def derive_gait(profile):

    height=profile["height_cm"]/100
    weight=profile["weight_kg"]
    fitness=profile["fitness"]

    leg_length=0.53*height
    stride_length=0.65*height

    speed=1.1+fitness*0.05

    cadence=speed/stride_length
    cadence=np.clip(cadence,1.4,2.2)

    vertical_acc=2.8+0.02*(weight-70)
    forward_acc=1.6+0.01*(fitness*10)
    lateral_acc=0.8+0.004*(height*100)

    asymmetry=np.clip((profile["age"]-40)/200,0,0.25)

    heel_strike=np.clip(0.6+0.002*(weight-70),0.5,1.0)

    return cadence,vertical_acc,forward_acc,lateral_acc,asymmetry,heel_strike


# --------------------------------------------------
# MIT timestamp simulation
# --------------------------------------------------

def generate_timestamps(avg_rows):

    start_ts=np.random.randint(22,39)
    end_ts=15000+np.random.randint(5,25)

    duration=end_ts-start_ts

    samples=int(np.random.normal(avg_rows,12))

    avg_gap=duration/(samples-1)

    gaps=np.random.normal(avg_gap,1.8,samples-1)
    gaps*=duration/np.sum(gaps)

    timestamps=np.zeros(samples)

    timestamps[0]=start_ts
    timestamps[1:]=start_ts+np.cumsum(gaps)

    timestamps=np.round(timestamps).astype(int)
    timestamps[-1]=end_ts

    return timestamps


# --------------------------------------------------
# Gait simulator
# --------------------------------------------------

def simulate(profile,avg_rows,gravity_axis,gravity_val):

    cadence,vertical_acc,forward_acc,lateral_acc,asym,heel=derive_gait(profile)

    timestamps=generate_timestamps(avg_rows)

    t=(timestamps-timestamps[0])/1000

    phase=2*np.pi*cadence*t

    vertical=vertical_acc*np.sin(phase)+0.3*vertical_acc*np.sin(2*phase)
    forward=forward_acc*np.sin(phase+np.pi/2)
    lateral=lateral_acc*np.sin(phase/2)

    ax=lateral
    ay=np.zeros_like(ax)
    az=forward

    if gravity_axis=="ay":
        ay=gravity_val+vertical
    elif gravity_axis=="az":
        az=gravity_val+vertical
    else:
        ax=gravity_val+vertical

    step_times=np.arange(0,t[-1],1/cadence)

    for i,st in enumerate(step_times):

        idx=np.searchsorted(t,st)

        if idx>=len(ax): break

        mod=1-asym if i%2 else 1

        for d in range(6):

            if idx+d<len(ax):

                if gravity_axis=="ay":
                    ay[idx+d]+=heel*1.2*np.exp(-d*0.7)*mod
                elif gravity_axis=="az":
                    az[idx+d]+=heel*1.2*np.exp(-d*0.7)*mod
                else:
                    ax[idx+d]+=heel*1.2*np.exp(-d*0.7)*mod

    wx=0.6*cadence*np.cos(phase)
    wy=0.4*cadence*np.sin(phase+0.3)
    wz=0.3*cadence*np.cos(phase/2)

    ax+=np.random.normal(0,0.05,len(ax))
    ay+=np.random.normal(0,0.05,len(ay))
    az+=np.random.normal(0,0.05,len(az))

    wx+=np.random.normal(0,0.15,len(wx))
    wy+=np.random.normal(0,0.15,len(wy))
    wz+=np.random.normal(0,0.15,len(wz))

    return pd.DataFrame({
    "timestamp":timestamps,
    "ax":ax,
    "ay":ay,
    "az":az,
    "wx":wx,
    "wy":wy,
    "wz":wz
    })


# --------------------------------------------------
# Main generation
# --------------------------------------------------

def main():

    avg_rows,gravity_axis,gravity_val=analyze_real_data()

    print("Detected gravity axis:",gravity_axis)

    with open(PROFILE_FILE) as f:
        profiles=json.load(f)

    os.makedirs(OUTPUT_DIR,exist_ok=True)

    for pid,p in profiles.items():

        folder=os.path.join(OUTPUT_DIR,pid.capitalize())
        os.makedirs(folder,exist_ok=True)

        for i in range(1,4):

            df=simulate(p,avg_rows,gravity_axis,gravity_val)

            df.round(4).to_csv(
            os.path.join(folder,f"{pid}_walk{i}.csv"),
            index=False
            )

        if int(pid.replace("person",""))%100==0:
            print("generated",pid)

if __name__=="__main__":
    main()