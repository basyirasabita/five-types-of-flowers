# Import Library
import streamlit as st
import pandas as pd
import numpy as np 
import warnings
warnings.filterwarnings('ignore')

# Library for visualization
import matplotlib.pyplot as plt 
import seaborn as sns 

# Function to plot numeric distribution (histogram & boxplot)
def plot_distribution(df, col, colName, color = 'indianred'):
    canvas = plt.figure(figsize=(12,5))
    
    # Plot Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(data=df[col], kde=True, color=color, bins=20)
    plt.title('Histogram')

    # Boxplot
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x=col, color=color)
    plt.title('Boxplot')

    plt.suptitle(f'{colName} Distribution')
    st.pyplot(canvas)

# Function to get the label and its count
def label_proportion(df):
    label_count = df.groupby('label')['label'].count().sort_values(ascending=False)

    labels = label_count.index.to_list()
    counts = label_count.to_list()

    return labels, counts

def run():
    # Set title
    st.title('5 Flower Types Classification Data Analysis')

    # Set subheader
    st.subheader('This page will display the EDA of the dataset')
    st.markdown('---')

    # Image
    st.image('./content/flower_banner2.JPEG', caption='Images taken from Unsplash')
    st.markdown('---')

    # Dataframe
    st.markdown('## 5 Flower Types Dataframe')
    data = pd.read_csv("./content/img.csv")
    dimension = pd.read_csv("./content/dimension.csv")
    st.dataframe(data.head(24))

    # EDA
    st.markdown('---')
    st.markdown('## Exploratory Data Analysis (EDA)')
    st.markdown('---')

    # Label Distribution 
    st.markdown('### Label Distribution')
    canvas = plt.figure(figsize=(2, 2))

    # Get label proportion of image dataframe
    labels, counts = label_proportion(data)

    # Save the label and counts into Dataframe
    label_df = pd.DataFrame({
        'label': labels,
        'count': counts
    })

    # Piechart
    plt.pie(label_df['count'], labels=labels, autopct='%.2f%%', startangle=90, textprops={'fontsize': 7})
    st.pyplot(canvas)

    st.markdown('We can see that each label accounts for 20% of the total. We observe that the data has balanced label counts. Since the label proportions are balanced, we can use accuracy as the metric.')

    st.markdown('---')

    # Display All Images
    st.markdown('### Display Images')
    st.image('./content/display_imgs.png')
    st.markdown("These are the samples of images that are used as the training data. We can see that the dataset consists of a multitude of variations such as different angles, color palettes, individual or clustered flowers, and varying levels of zoom present across the images.")
    st.markdown('---')

    # Plot Distribution
    distribution = { 
        'size_kb': '', 
        'width': '',
        'height': ''
    }
    # Select the column
    st.markdown('### Image Properties Distribution')
    column = st.selectbox(
        'Choose Column',
        [ 
            'Size (kb)', 
            'Width (px)',
            'Height (px)'
        ],
        index=None,
        placeholder='Choose Column'
    )

    if column == None:
        st.markdown('Please Choose the Column First to See Visualization')
    elif column == 'Size (kb)':
        st.dataframe(data.describe().T)
        st.markdown('We observe that the average size of all the images is 51.075 kb. The maximum size of all the images is 3426.434 kb, and the minimum size of all images is 2.936 kb. The difference between the maximum and minimum sizes is quite extensive. Therefore, we need to see the differences between the small and large images.')
        plot_distribution(data, 'size_kb', column, 'indianred')
        st.markdown('As seen in the visualization, the size of the images follows a skewed distribution, with several images having significantly larger sizes compared to the overall distribution. The majority of images fall within a size range lower than 500 kb.')
    elif column == 'Width (px)':
        st.dataframe(dimension['width'].describe().to_frame().T)
        st.markdown('We observe that the mean width of the images is 505.2606 pixels, with the smallest width among the images being 101 pixels and the largest width 4987 pixels. This extensive difference indicates a wide variation in the resolutions of the image dataset.')
        plot_distribution(dimension, 'width', column, 'darksalmon')
        st.markdown("As seen in the visualization, the width of the images follows a skewed distribution. There are quite a lot of outliers observed in the boxplot, indicating that several images have an extensive width compared to the overall dataset. The majority of image widths fall below 1000 pixels, which is considered sufficient and decent for training purposes.")
    else:
        st.dataframe(dimension['height'].describe().to_frame().T)
        st.markdown('We observe that the mean height of the images is 424.4654 pixels, with the smallest height among the images being 113 pixels and the largest height 4579 pixels. This significant difference further supports the indication of a wide variation in the resolutions of the image dataset.')
        plot_distribution(dimension, 'height', column, 'gold')
        st.markdown('As seen in the visualization, the height of the images follows a skewed distribution. There are quite a lot of outliers observed in the boxplot, indicating that several images have an extensive height compared to the overall dataset. The majority of image heights fall below 1000 pixels, which is considered sufficient and decent for training purposes.')

    st.markdown('---')

    # Display Small Images
    st.markdown('### Display Small Images')
    st.image('./content/small_img_samples.png')
    st.markdown('These are the samples of images that are smaller than 5 kb. We can see that there are no specific characteristics seen in the small image. We can infer that the small image just indicates the smaller resolution of the images.')
        
    st.markdown('---')

    # Display Large Images
    st.markdown('### Display Large Images')
    st.image('./content/large_img_samples.png')
    st.markdown('These are the samples of images that are larger than 1500 kb. We can observe that there are no specific characteristics apparent in the larger images. Therefore, we can infer that the larger size primarily indicates higher resolution rather than any distinctive features.')
        
    st.markdown('---')
    st.text('Basyira Sabita - 2024')
if __name__=='__main__':
    run()