import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
from io import StringIO
from IPython.display import clear_output
import time
st.set_page_config(layout="wide")
st.header('K Means Clusters')
#Generating Random Dataframe
@st.cache_data
def generate_random_dataframe():
    x=np.random.randint(100,1000,1500)
    y=np.random.randint(100,1000,1500)
    x1=np.random.randint(100,1000,1500)
    y1=np.random.randint(100,1000,1500)
    x2=np.random.randint(100,1000,1500)
    y2=np.random.randint(100,1000,1500)
    dct={'X':list(x)+list(x1)+list(x2),'y':list(y)+list(y1)+list(y2)}
    return dct

#Creating Dataframe
dct=generate_random_dataframe()
df=pd.DataFrame(dct)

color_scale = st.selectbox('Select Color Palette:', px.colors.named_colorscales(), index=0)

#Selecting k random centroids
k= st.selectbox("Select number of clusters",[1,3,5,7,9])
button_clicked = st.button('NEXT')
reset_flag = st.button('RESET')
def get_centroids(data,k):
    lst=[]
    for i in range(1,k+1):
        lst.append(data.apply(lambda x: int(x.sample())))
    return pd.concat(lst,axis=1)

#This will calculate euclidean distance from random centroids to all the existing data points.
# and assigns Labels for the existing data points based on the random Centroids
def get_labels(centroids,df):
    distances=centroids.apply(lambda x:np.sqrt(((x-df)**2).sum(axis=1)))
    
    return distances.idxmin(axis=1)

#Calculating geometric mean of all the data points from each group to get new centroids
def get_new_centroids(df,labels,k):
    return df.groupby(labels).apply(lambda x:np.exp((np.log(x)).mean())).T

#for plotting the clusters formation

def plot_clusters(df,labels,centroid,k,iteration):
    clear_output(wait=True)
    # plt.title(f'Iteration {iteration}')
    # plt.scatter(x=df.iloc[:,0],y=df.iloc[:,1],c=labels)
    trace0 = go.Scatter(x=df.iloc[:,0], y=df.iloc[:,1],mode='markers',marker=dict(color=df['labels'],colorscale=color_scale))
    #fig1=px.scatter(df,x=df.iloc[:,0],y=df.iloc[:,1],color=labels)
    go.Scatter()
    # plt.scatter(centroid.T.iloc[:,0],centroid.T.iloc[:,1],c='r')
    trace1 = go.Scatter(x=centroid.T.iloc[:,0], y=centroid.T.iloc[:,1],mode='markers',marker=dict(color='red',size=18))
    fig = go.Figure([trace0, trace1])
    st.plotly_chart(fig)
    #fig2=px.scatter(centroid,x=centroid.T.iloc[:,0],y=centroid.T.iloc[:,1],color=range(1,6))
    #fig1.show()
    #fig2.show()
    # plt.show()
   



# import streamlit as st
# import matplotlib.pyplot as plt
# import pandas as pd
# from IPython.display import clear_output
# max_iteration=100
# n=1
# k=5

# def plot_clusters(df, labels, centroid, k, iteration):
#     centroid1 = centroid.T
#     df1 = df.copy()

#     # Clear the output for a cleaner display
#     clear_output(wait=True)

#     # Create an empty container
#     plot_container = st.empty()

#     with plot_container.container():
#         # Create a Matplotlib figure
#         fig, ax = plt.subplots()
#         # Set the title for the plot
#         ax.set_title(f'Iteration {iteration}')

#         # Scatter plot for data points
#         ax.scatter(df1['X'], df1['y'], c=labels, cmap='viridis', alpha=0.5, label='Data Points')

#         # Scatter plot for centroids
#         ax.scatter(centroid1['X'], centroid1['y'], c='r', marker='X', s=100, label='Centroids')

#         # Display legend
#         ax.legend()

#         # Display the Matplotlib plot in the Streamlit app
#         plot_container.pyplot(fig)
#     plot_container.empty()

# # # Example usage:
# # # Assuming df, labels, centroid, k are defined
# # # Assuming 'X' and 'y' are the correct column names in your DataFrame
# # for iteration in range(1, 16):
# #     plot_clusters(df, labels, centroid, k, iteration)



    #-----Main code-----#

max_iteration=100
n=1
centroids=get_centroids(df,k)
old_centroids=pd.DataFrame()
# while n<max_iteration and not old_centroids.equals(centroids):
#     old_centroids=centroids
#     #print(old_centroids)
#     labels=get_labels(centroids,df)
#     #print(labels)
#     df['labels']=labels
#     centroids=get_new_centroids(df,labels,k)
#     #print(centroids)
#     plot_clusters(df,labels,centroids,k,n)
#     time.sleep(1)
#     n+=1



# if button_clicked:
#     old_centroids=centroids
#     #print(old_centroids)
#     labels=get_labels(centroids,df)
#     #print(labels)
#     df['labels']=labels
#     centroids=get_new_centroids(df,labels,k)
#     #print(centroids)
#     plot_clusters(df,labels,centroids,k,n)

def get_session():
    return st.session_state.setdefault('session', {'centroids': None, 'labels': None, 'df':None,'reset_flag': False})
def calculate_clusters():
    session = get_session()

    old_centroids = session['centroids']
    if old_centroids is None:
        old_centroids = get_centroids(df, k)  # Replace with your initialization function
    
    labels = get_labels(old_centroids, df)
    df['labels'] = labels
    centroids = get_new_centroids(df, labels, k)
    
    
    if centroids.equals(session['centroids']):
        st.write('Final Clusters have been formed.')
        plot_clusters(df, labels, centroids, k, n)
        st.write('The Centroids are:')
        st.write(centroids.T.iloc[:,:2])
        
    else:
        
        plot_clusters(df, labels, centroids, k, n)
        session['centroids'] = centroids
        session['labels'] = labels
    


session=get_session()
if button_clicked:
    calculate_clusters()

if reset_flag:
    # Set the reset flag in the session state
    session['reset_flag'] = True
    session['centroids'] = None
    session['labels'] = None
    
# if session['reset_flag']:
#     # If reset flag is set, allow user to select a new number of clusters
#     session['reset_flag'] = False  # Reset the flag
#     calculate_clusters()
#     session['centroids'] = centroids
#     session['labels'] = labels
    
    
        
        
