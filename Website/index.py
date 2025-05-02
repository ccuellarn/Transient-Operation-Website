from flask import Flask, render_template, request, redirect, url_for
from datetime import datetime
import numpy as np
import pandas as pd
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, Angle
import matplotlib.pyplot as plt
import io
import base64
from io import BytesIO
from matplotlib import cm
from astropy.visualization import astropy_mpl_style, quantity_support

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    datos_tabla = None
    plot_url = None
    plot_url_altaz = None 
    alertas = None
    error = None
    
    if request.method == 'POST':
        try:
        # Obtener parámetros del formulario
            latitud = float(request.form.get('latitud'))
            longitud = float(request.form.get('longitud'))
            fecha_inicio = request.form.get('fecha_inicio')
            fecha_fin = request.form.get('fecha_fin')
            escala_tiempo_unidad = request.form.get('escala_tiempo_unidad')
            escala_tiempo_valor = int(request.form.get('escala_tiempo_valor'))
            limite_altitud = int(request.form.get('limite_altitud'))
            magnitud_minima = float(request.form.get('magnitud_minima'))
            parametro_k = float(request.form.get('parametro_k'))
        
            modo_ingreso = request.form.get('modo_ingreso', 'manual')

            if modo_ingreso == 'csv' and 'csv_file' in request.files:
                # Procesar archivo CSV
                csv_file = request.files['csv_file']
                if csv_file.filename != '':
                    # Leer CSV en DataFrame
                    df = pd.read_csv(io.StringIO(csv_file.read().decode('utf-8')))
                    
                    # Validar columnas
                    required_cols = ['nombre', 'ra', 'dec', 'magnitud']
                    if not all(col in df.columns for col in required_cols):
                        raise ValueError("El CSV debe contener las columnas: nombre, ra, dec, magnitud")
                    
                    objetos = df.to_dict('records')

        # Obtener alertas de objetos
            else: 
                nombres = request.form.getlist('nombre_objeto[]')
                ras = request.form.getlist('ra_objeto[]')
                decs = request.form.getlist('dec_objeto[]')
                mags = request.form.getlist('mag_objeto[]')
        
                alertas = {
            'Name': nombres,
            'RA': ras,
            'DEC': decs,
            'Mag': [float(mag) for mag in mags]
                }

        except Exception as e:
            error = str(e)

        # Preparar parámetros para las funciones
        observer = (latitud, longitud)
        date_i = fecha_inicio.replace('T', ' ') + ':00'
        date_f = fecha_fin.replace('T', ' ') + ':00'
        timescale = [escala_tiempo_unidad, escala_tiempo_valor]
        m_min = magnitud_minima
        K = parametro_k
        many = len(alertas['Name']) + 1
        
        # Llamar a las funciones de cálculo
        data_observations = Observations(
            observer=observer,
            alert=pd.DataFrame(alertas),
            Date_i=date_i,
            Date_f=date_f,
            time_scale=timescale,
            rango=many,
            m_min=m_min,
            limit=limite_altitud,
        )
        
        time_array = CreateTime(date_i, date_f, timescale)
        general_time = Time(date_i, format='iso', scale='utc') + np.arange(len(time_array)) * (
            escala_tiempo_valor * u.minute if escala_tiempo_unidad == 'm' else 
            escala_tiempo_valor * u.second if escala_tiempo_unidad == 's' else 
            escala_tiempo_valor * u.hour
        )
        
        order = Order(
            Data=data_observations,
            rango=many,
            Time=time_array,
            General_time=general_time,
            K=K
        )
        
        graphic = Graphic(Data=data_observations,
                    rango=many,
                    time=time_array)

        if isinstance(order, pd.DataFrame) and not order.empty:
            # Convertir el DataFrame a un formato adecuado para la plantilla
            datos_tabla = []
            for _, row in order.iterrows():
                datos_tabla.append({
                    'hora': Time(row['Time'], format='iso').iso.split()[1][:8],
                    'label': row['Label'],
                    'objeto': row['Name'],
                    'ra': row['RA'],
                    'dec': row['DEC'],
                    'magnitud': f"{row['Mag']:.2f}",
                    'tiempo_exposicion': f"{row['Time expo']:2f}s"

                })
            names = [i for i in order['Name'].drop_duplicates()]
            ra = [Angle(i, unit=u.deg).degree for i in order['RA'].drop_duplicates()]
            dec = [Angle(i, unit=u.deg).degree for i in order['DEC'].drop_duplicates()]

            plt.figure(figsize=(9, 7))

            plt.title('Starting time {}\n Location: ({} , {})'.format(date_i,observer[0],observer[1]))

            plt.scatter(ra,dec,color='purple',marker='*', s=80)
            for i in range(0, len(dec)):
                plt.text(ra[i], dec[i], '{}) {}'.format(str(i+1), str(names[i])), color='k')
    
            plt.ylabel(r"Declination [deg]",fontsize=14)
            plt.xlabel(r'Right ascension [deg]',fontsize=14)
            plt.grid(True)
            plt.tight_layout()
            # # Generar la gráfica
            # plt.figure(figsize=(11,7))

            # ordertime = []
            # min_alt = []
            # max_alt = []
            # orderlabels = [i for i in order['Label'].drop_duplicates()]

            # for i in orderlabels: 
            #     hour_i = order.loc[order['Label']==i].head(1)['Time'].iloc[0].datetime.hour
            #     hour_f = order.loc[order['Label']==i].tail(1)['Time'].iloc[0].datetime.hour
            #     minute_i = order.loc[order['Label']==i].head(1)['Time'].iloc[0].datetime.minute
            #     minute_f = order.loc[order['Label']==i].tail(1)['Time'].iloc[0].datetime.minute

    
            #     if 13 <= hour_i <= 23:
            #         hour_i = hour_i-24
            #     if 13 <= hour_f <= 23:
            #         hour_f = hour_f-24
        
            #     ordertime.append((float(f"{hour_i}.{minute_i}"),float(f"{hour_f}.{minute_f}"))) 

            
            # for each in graphic:
            #     if each['Label'].iloc[0] in orderlabels:

            #         ALT = each['Alt'].tolist()
            #         AZ = each['Az'].tolist()
            #         T = each['Time'].tolist()

            #         plt.scatter(T, ALT,c=AZ,cmap="viridis",lw=1)

            #         for j, alt in enumerate(ALT):
            #             plt.annotate(str(each['Label'].iloc[0]), (T[j], alt + 1), color='k')        

            #         min_alt.append(min(ALT))
            #         max_alt.append(max(ALT))

            # # Limits of good observations
            # plt.plot(time_array, [limite_altitud] * len(time_array), '--', color='grey')
            # #Details
            # plt.colorbar().set_label("Azimuth [deg]")
            # plt.xlabel("Hours around Midnight")
            # plt.ylabel("Altitude [deg]")

            # # Limits of good observations
            # for each in ordertime:
            #     plt.plot([each[0]] * len(np.linspace(min(min_alt),max(max_alt), len(time_array))), np.linspace(min(min_alt),max(max_alt), len(time_array)), '--', color='orange')
            #     plt.plot([each[1]] * len(np.linspace(min(min_alt),max(max_alt),len(time_array))), np.linspace(min(min_alt),max(max_alt), len(time_array)), '--', color='salmon')

            #     plt.text( each[0],min(min_alt), 'S', color='orange')
            #     plt.text( each[1],min(min_alt), 'F',color='salmon')

            # #Exes
            # hours = [str(round(num + 24,1)) if num < 0 else str(round(num,1)) for num in np.array([-6,-4,-2,0,2,4,6])]
            # plt.xticks(np.array([-6,-4,-2,0,2,4,6]), hours)
            # plt.title('Starting time {}\n Location: ({} , {})'.format(date_i,observer[0],observer[1]))
            # plt.grid(True)

            # Convertir la gráfica a imagen base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plot_url = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()  # Cerrar la figura para liberar memoria


    return render_template('planificador_astro.html',
                        error=error, 
                         datos_tabla=datos_tabla,
                         plot_url=plot_url,
                         plot_url_altaz=plot_url_altaz,
                         año_actual=datetime.now().year)


# Funciones astronómicas proporcionadas
def DeltaTime(Date_i, Date_f, t_scale):
    scale = t_scale[0]
    sc = t_scale[1]

    t_i = Time(Date_i,format = 'iso', scale='utc')
    t_f = Time(Date_f,format = 'iso', scale='utc')
    dif_sec = (t_f - t_i).sec

    if scale == 's':
        len = int(dif_sec / sc) + 1
        t_ = np.linspace(t_i.datetime.hour-24 , t_f.datetime.hour , len) 
   
    if scale == 'm':
        len = int(dif_sec / (sc*60)) + 1
        t_ = np.linspace(t_i.datetime.hour-24 , t_f.datetime.hour , len) 
    
    if scale == 'h':
        len = int(dif_sec / (sc*3600)) + 1
        t_ = np.linspace(t_i.datetime.hour-24 , t_f.datetime.hour , len) 
    
    return t_

#Graphic
def Graphic(Data,rango,time):
    targets = []

    for i in range(1,rango):
        Target = Data.loc[Data['Label'] == i].copy()
        if Target.empty:
            continue
        
        Target['Time'] = time
        Target = Target[Target['Observable'] != False]

        if not Target.empty:
            targets.append(Target)

    if not targets:
        return [], pd.DataFrame()
    
    return targets

def CreateTime(Date_i, Date_f, t_scale):
    time_midnight = Time(Time(Date_i, format='iso', scale='utc').iso.split()[0] + ' 00:00:00', 
                        format='iso', scale='utc')
    delta = DeltaTime(Date_i, Date_f, t_scale)*u.hour
    return time_midnight + delta

def Observations(observer, alert, Date_i, Date_f, time_scale, rango, m_min,limit):
    time = CreateTime(Date_i, Date_f, time_scale)
    lat_conv, lon_conv = observer
    observer_loc = EarthLocation(lat=lat_conv*u.deg, lon=lon_conv*u.deg)
    Big_Data = []
    
    alert['Label'] = range(1, rango)
    alert = alert[alert['Mag'] <= m_min].copy()

    for each_time in time:
        celestial_coord = SkyCoord(ra=alert['RA'], dec=alert['DEC'])
        altaz_coord = celestial_coord.transform_to(AltAz(obstime=each_time, location=observer_loc))
        state = altaz_coord.alt > limit*u.deg
        
        alert['Observable'] = state
        alert['Az'] = altaz_coord.az.deg
        alert['Alt'] = altaz_coord.alt.deg
        
        Data = alert.copy()
        Big_Data.append(Data)

    return pd.concat(Big_Data, axis=0)

def ExpositionTime(K, m):
    return K*(10**(0.4*(m)))

def Order(Data, rango, Time, General_time, K):
    targets = []
    order = []
    order_expo =[]
    finalorder = []

    if Data.empty:
        return 'No se encontraron observaciones'

    for i in range(1, rango):
        Target = Data.loc[Data['Label'] == i].copy()
        if Target.empty:
            continue
        
        Target['Time'] = Time
        Target['Time expo'] = ExpositionTime(K, Target['Mag'])
        Target = Target[(Target['Observable'] != False) & (Target['Time expo'].between(1, 3600, inclusive='neither'))]

        if not Target.empty:
            targets.append(Target)

    if not targets:
        return pd.DataFrame()

    obs = pd.concat(targets, axis=0).reset_index(drop=True).drop(['Observable'], axis=1)

    for i in range(0,len(Time)):
        priority = obs.loc[obs['Time'] == Time[i]].sort_values('Alt', ascending=False, na_position='first').head(1)
        priority['Time'] = General_time[i]
        order.append(priority)

    order = pd.concat(order,axis=0).reset_index(drop=True).drop(['Az'],axis=1)
    orderlabels = [i for i in order['Label'].drop_duplicates()]

    for each in orderlabels:
        partial_order = order.loc[order['Label'] == each]
        best_alt = partial_order.sort_values('Alt', ascending=False, na_position='first').head(1)
        expo_min = best_alt['Time'].iloc[0] - int(partial_order['Time expo'].iloc[0]*0.5) * u.second
        expo_max = best_alt['Time'].iloc[0] + int(partial_order['Time expo'].iloc[0]*0.5) * u.second 

        order_expo = partial_order[partial_order['Time'].between(expo_min, expo_max, inclusive='neither')]

        if len(order_expo) == 1:
            index = order_expo.index
            min_index = partial_order.head(1).index
            max_index = partial_order.tail(1).index
            final_index = order.tail(1).index

            if not (index == max_index):
                order_expo = pd.concat([order_expo, partial_order.loc[index+1]],axis=0).reset_index(drop=True)
                finalorder.append(order_expo)

            if (index == final_index) and (index != min_index):
                order_expo = pd.concat([partial_order.loc[index-1],order_expo],axis=0).reset_index(drop=True)
                finalorder.append(order_expo)   

        else:
            order_expo = order_expo.reset_index(drop=True)
            finalorder.append(order_expo)

    finalorder = pd.concat(finalorder,axis=0).reset_index(drop=True).drop(['Alt'],axis=1)
    
    return finalorder


if __name__ == '__main__':
    app.run(debug=True)