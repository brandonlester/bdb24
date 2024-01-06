#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!jupyter nbconvert --to script local_functions.ipynb


# In[ ]:


def scale_column_to_100(dataframe, column_name):

    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler


    # Check if the column exists in the DataFrame
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    # Extract the column to be scaled
    column_to_scale = dataframe[column_name].values.reshape(-1, 1)

    # Initialize MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 100))

    # Fit and transform the column
    scaled_column = scaler.fit_transform(column_to_scale)

    # Update the DataFrame with the scaled column
    dataframe[column_name] = scaled_column

    return dataframe


# In[ ]:


def extract_position(s):

    import re
    
    match = re.search(r'\d', s)
    if match:
        return s[:match.start()]
    else:
        return s

import numpy as np
extract_position_vectorized = np.vectorize(extract_position)


# ### Tracking Data Standardization

# In[5]:


def create_field_grid():

    import pandas as pd
    import numpy as np
    from itertools import product

    # Define the values for each variable using numpy.linspace
    x_values = np.linspace(0, 120, num=120)
    y_values = np.linspace(0, 160/3, num=int(160/3))

    # Generate the Cartesian product using itertools.product
    cartesian_product = list(product(x_values, y_values))

    # Convert the result to a DataFrame
    df = pd.DataFrame(cartesian_product, columns=['x', 'y'])

    return df


# ### Distance Calculations

# In[1]:


def euclidean_distance_df(df, x1, y1, x2, y2, ref_name):

    import numpy as np

    df['dist_to_' + ref_name] = np.sqrt((df[x1] - df[x2])**2 + (df[y1] - df[y2])**2)

    return df


# In[ ]:


def calc_dist_1frame(frame):

    import pandas as pd
    from scipy.spatial.distance import cdist

    # make unique positions, as to not duplicate columns based on player position
    frame['pos_unique'] = (frame['position']
                        .add(frame
                            .groupby('position', as_index=False)
                            .cumcount()
                            .add(1)
                            .dropna()
                            .astype(str)
                            .str.replace('.0','', regex=False)
                            .str.replace('0','', regex=False)))

    # calc distances 
    _df = (pd
        .DataFrame(cdist(frame.loc[:, ['x_std', 'y_std']], 
                        frame.loc[:, ['x_std', 'y_std']]), 
                    index=frame['nflId'], 
                    columns=frame['pos_unique'].fillna('football')))

    # reset index to pop out nflId into its own column
    _df = _df.reset_index()

    # merge new distance values onto original dataframe
    frame = frame.merge(_df)

    return frame

def calc_dist_1play(play):

    import pandas as pd

    df_all_frames = pd.DataFrame()

    for fid in play['frameId'].unique():

        df_frame = play.loc[play['frameId']==fid].copy()

        df_frame_dists = calc_dist_1frame(df_frame)

        # concatenate new results into the output dataframe 
        df_all_frames = pd.concat([df_all_frames, df_frame_dists])

    return df_all_frames

""" game = tracking[(tracking["gameId"] == 2022090800)]

df_all_plays = pd.DataFrame()

for pid in game["playId"].unique():

    df_play = game.loc[game['playId']==pid].copy()

    df_play_dists = calc_dist_1play(df_play)

    df_all_plays = pd.concat([df_all_plays, df_play_dists])

df_all_plays.head() """


# In[ ]:


# difference in orientation in degrees between the way the player is facing and where the reference player is facing
# 0 is facing directly at the reference player, 180 is directly away

def calc_angle_diff(input_df, xc, yc, anglec, xc_ref, yc_ref, new_name_suffix):

    import numpy as np
    import pandas as pd
    
    df = input_df.copy()
    xdist_col = 'x_dist_to_' + new_name_suffix
    ydist_col = 'y_dist_to_' + new_name_suffix
    dist_col = 'dist_to_' + new_name_suffix

    df[xdist_col] = df[xc_ref] - df[xc]
    df[ydist_col] = df[yc_ref] - df[yc]

    df[dist_col] = np.sqrt(np.square(df[xdist_col]) + np.square(df[ydist_col]))

    df['tmp'] = np.arctan2(df[ydist_col], df[xdist_col]) * (180 / np.pi)

    df['tmp'] = (360 - df['tmp']) + 90

    df['tmp'] = np.where(df['tmp'] < 0, 
                        df['tmp'] + 360,
                        np.where(df['tmp'] > 360, 
                                df['tmp'] - 360, 
                                df['tmp']))

    df['diff'] = np.abs(df[anglec] - df['tmp'])

    df[anglec + '_to_' + new_name_suffix] = np.minimum(360 - df['diff'], df['diff'])

    df = df.drop(['tmp', 'diff'], axis=1)

    return df


# # Field Control

# In[ ]:


def compute_rotation_matrix(v_theta):

    import numpy as np

    R = np.array([[np.cos(v_theta), -np.sin(v_theta)],
                  [np.sin(v_theta), np.cos(v_theta)]])
    return R

def compute_scaling_matrix(radius_of_influence, s_ratio):

    import numpy as np

    S = np.array([[radius_of_influence * (1 + s_ratio), 0],
                  [0, radius_of_influence * (1 - s_ratio)]])
    return S

def compute_covariance_matrix(v_theta, radius_of_influence, s_ratio):

    import numpy as np

    R = compute_rotation_matrix(v_theta)
    S = compute_scaling_matrix(radius_of_influence, s_ratio)
    Sigma = np.dot(np.dot(R, S), np.dot(S, np.linalg.inv(R)))
    return Sigma


# In[ ]:


# note that this is meant operate on just 1 row of the tracking dataset
def compute_player_zoi(player_frame_tracking_data, input_field_grid):

    import pandas as pd
    import numpy as np
    from scipy.stats import multivariate_normal

    if input_field_grid is None:
        field_grid = create_field_grid()
    else:
        field_grid = input_field_grid.copy()
        
    # gid = player_frame_tracking_data['gameId']
    # pid = player_frame_tracking_data['playId']
    # fid = player_frame_tracking_data['frameId']
    # nid = player_frame_tracking_data['nflId']

    zoi_center_x_ = player_frame_tracking_data['x_next']
    zoi_center_y_ = player_frame_tracking_data['y_next']
    v_theta_ = player_frame_tracking_data['v_theta']
    radius_of_influence_ = player_frame_tracking_data['radius_of_influence']
    s_ratio_ = player_frame_tracking_data['s_ratio']

    mu = np.array([zoi_center_x_, zoi_center_y_])
    Sigma = compute_covariance_matrix(v_theta_, radius_of_influence_, s_ratio_)

    mvn = multivariate_normal(mean=mu, cov=Sigma)
    influence = mvn.pdf(field_grid[['x', 'y']])
    influence /= np.max(influence)

    # player_zoi = field_grid.assign(
    #     gameId = gid,
    #     playId = pid,
    #     frameId = fid,
    #     nflId = nid,
    #     influence=influence
    # )

    # return player_zoi

    return influence


# The barycentric coordinate system is a way to represent a point in a triangle using weights. For a point P(x, y) in a triangle ABC, the barycentric coordinates (u, v, w) are such that:
# 
# P = u * A + v * B + w * C

# In[ ]:


def project_triangle(x, y, facing_angle, yd, xd, maxX, maxY, minX, minY):

    import math
    import pandas as pd
    import numpy as np

    # Calculate base endpoint
    x_base = x + yd * math.cos(math.radians(facing_angle))
    y_base = y + yd * math.sin(math.radians(facing_angle))

    # Calculate perpendicular sides endpoints
    x_perp1 = x_base - xd * math.sin(math.radians(facing_angle))
    y_perp1 = y_base + xd * math.cos(math.radians(facing_angle))

    x_perp2 = x_base + xd * math.sin(math.radians(facing_angle))
    y_perp2 = y_base - xd * math.cos(math.radians(facing_angle))

    # Check and adjust for limits
    x_perp1 = min(maxX, max(minX, x_perp1))
    y_perp1 = min(maxY, max(minY, y_perp1))

    x_perp2 = min(maxX, max(minX, x_perp2))
    y_perp2 = min(maxY, max(minY, y_perp2))

    return [(x_perp1, y_perp1), (x_base, y_base), (x_perp2, y_perp2)]




def is_point_in_triangle(p, a, b, c):

    """
    Check if a point is inside a triangle using barycentric coordinates.

    Parameters:
    - p: Point to check (tuple of x, y)
    - a, b, c: Vertices of the triangle (tuples of x, y)

    Returns:
    - True if the point is inside the triangle, False otherwise
    """
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    # Normalize vertices and point
    a_norm = (a[0] / max(a), a[1] / max(a))
    b_norm = (b[0] / max(b), b[1] / max(b))
    c_norm = (c[0] / max(c), c[1] / max(c))
    p_norm = (p[0] / max(p), p[1] / max(p))

    d1 = sign(p_norm, a_norm, b_norm)
    d2 = sign(p_norm, b_norm, c_norm)
    d3 = sign(p_norm, c_norm, a_norm)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


# In[ ]:


# Calculate
def line_equation(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    # Calculate the slope
    if x2 - x1 != 0:
        m = (y2 - y1) / (x2 - x1)
    else:
        # Avoid division by zero if the points have the same x-coordinate
        raise ValueError("The x-coordinates of the points cannot be the same")

    # Calculate the y-intercept
    b = y1 - m * x1

    # Return the equation of the line
    return f"y = {m}x + {b}"


def perpendicular_projection(point, line_equation):
    x0, y0 = point

    # Parse the equation to extract slope (m) and y-intercept (b)
    parts = line_equation.split()
    m = float(parts[2][:-1])  # Extracting the slope, excluding the 'x'
    b = float(parts[-1])      # Extracting the y-intercept

    # Calculate the perpendicular projection
    xp = (m*x0 + y0 - m*b) / (m**2 + 1)
    yp = m * xp + b

    return xp, yp


# ### Play Animation

# In[ ]:


# # libraries

# from scipy.stats import kde
 
# # create data
# x = df_control_1frame['x']
# y = df_control_1frame['y']
 
# # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
# nbins=300
# k = kde.gaussian_kde([x,y])
# xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
# zi = k(np.vstack([xi.flatten(), yi.flatten()]))
 
# # Make the plot
# plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
# plt.show()
 
# # Change color palette
# plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap=plt.cm.Greens_r)
# plt.show()


# In[1]:


def create_football_field(
    linenumbers=True,
    endzones=True,
    figsize=(12, 6.33),
    line_color="black",
    field_color="white",
    ez_color=None,
    ax=None,
    return_fig=False,
):
    """
    Function that plots the football field for viewing plays.
    Allows for showing or hiding endzones.
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from IPython.display import display, HTML
    import matplotlib.patches as patches
    import math


    if ez_color is None:
        ez_color = field_color

    rect = patches.Rectangle(
        (0, 0),
        120,
        53.3,
        linewidth=0.1,
        edgecolor="r",
        facecolor=field_color,
        zorder=0,
    )

    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    ax.add_patch(rect)
    ax.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             color=line_color)
    
    # Endzones
    if endzones:
        ez1 = patches.Rectangle(
            (0, 0),
            10,
            53.3,
            linewidth=0.1,
            edgecolor=line_color,
            facecolor=ez_color,
            alpha=0.6,
            zorder=0,
        )
        ez2 = patches.Rectangle(
            (110, 0),
            10,
            53.3,
            linewidth=0.1,
            edgecolor=line_color,
            facecolor=ez_color,
            alpha=0.6,
            zorder=0,
        )
        ax.add_patch(ez1)
        ax.add_patch(ez2)

    ax.axis("off")
    if linenumbers:
        for x in range(20, 110, 10):
            numb = x
            if x > 50:
                numb = 120 - x
            ax.text(
                x,
                5,
                str(numb - 10),
                horizontalalignment="center",
                fontsize=20,  # fontname='Arial',
                color=line_color,
            )
            ax.text(
                x - 0.95,
                53.3 - 5,
                str(numb - 10),
                horizontalalignment="center",
                fontsize=20,  # fontname='Arial',
                color=line_color,
                rotation=180,
            )
    if endzones:
        hash_range = range(11, 110)
    else:
        hash_range = range(1, 120)

    for x in hash_range:
        ax.plot([x, x], [0.4, 0.7], color=line_color)
        ax.plot([x, x], [53.0, 52.5], color=line_color)
        ax.plot([x, x], [22.91, 23.57], color=line_color)
        ax.plot([x, x], [29.73, 30.39], color=line_color)

    border = patches.Rectangle(
        (-5, -5),
        120 + 10,
        53.3 + 10,
        linewidth=0.1,
        edgecolor="orange",
        facecolor=line_color,
        alpha=0,
        zorder=0,
    )
    ax.add_patch(border)
    ax.set_xlim((-5, 125))
    ax.set_ylim((-5, 53.3 + 5))

    if return_fig:
        return fig, ax
    else:
        return ax


# In[2]:


def animate_tracking_data(tracking_df, id_game_play, x_col, y_col, dir_col, dir_arrow_metric, o_col):

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from IPython.display import display, HTML
    import matplotlib.patches as patches
    import math

    n = tracking_df[tracking_df.game_play_id == id_game_play].frameId.max()

    # Initialize the football field plot
    fig, ax = create_football_field(return_fig=True)

    # Get unique club names and assign colors
    clubs = tracking_df[(tracking_df.game_play_id == id_game_play) & (tracking_df['club'] != 'football')]['club'].unique()
    print(clubs)
    club_colors = {clubs[0]: 'orange', clubs[1]: 'lightblue', 'football': 'brown'}
    
    texts = []  # To store jersey number text elements
    dir_arrows = []
    o_arrows = []

    # Initialize the scatter plot for each club.
    scatters = {}
    for club in tracking_df.club.unique():
        color = club_colors.get(club, 'white')
        if club == "football":
            scatters[club] = ax.scatter([], [], label=club, s=80, color=color, lw=1, edgecolors="black", zorder=5)
        else:
            scatters[club] = ax.scatter([], [], label=club, s=170, color=color, lw=1, edgecolors="black", zorder=5)
            
    ax.legend().remove()

    def update(frame):
        # Clear previous frame's texts
        for text in texts:
            text.remove()
        texts.clear()

        for arrow in dir_arrows:
            arrow.remove()
        dir_arrows.clear()

        for arrow in o_arrows:
            arrow.remove()
        o_arrows.clear()
        

        frame_data = tracking_df[(tracking_df.game_play_id == id_game_play) & (tracking_df.frameId == frame)]
        event_for_frame = frame_data['event'].iloc[0]  # Assuming each frame has consistent event data
        if pd.notna(event_for_frame):
            ax.set_title(f"Tracking data for {id_game_play}: at frame {frame}\nEvent: {event_for_frame}", fontsize=15)
        else:
            ax.set_title(f"Tracking data for {id_game_play}: at frame {frame}", fontsize=15)

        for club, d in frame_data.groupby("club"):
            scatters[club].set_offsets(np.c_[d[x_col].values, d[y_col].values])
            scatters[club].set_color(club_colors.get(club, 'white'))
            scatters[club].set_edgecolors("black")  # Explicitly setting the edge color
            
            # Display jersey numbers if it's not the football
            if club != "football":
                for _, row in d.iterrows():
                    text = ax.text(row[x_col], row[y_col], str(int(row["jerseyNumber"])), 
                                   fontsize=8, ha='center', va='center', color="black", fontweight='bold', zorder=6)
                    texts.append(text)
                
                for index, row in d.iterrows():
                    try:
                        
                        #direction arrows
                        dir_angle = math.radians(90-row[dir_col])

                        dir_dx = row[dir_arrow_metric] * math.cos(dir_angle)
                        dir_dy = row[dir_arrow_metric] * math.sin(dir_angle)

                        dir_arrow = ax.quiver(row[x_col], row[y_col], dir_dx, dir_dy, angles='xy', scale_units='xy', width = 0.004, scale=1, alpha=0.5, color = 'orange')
                        dir_arrows.append(dir_arrow)

                        #orientation arrows
                        o_angle = math.radians(90-row[o_col])

                        o_dx = 3 * math.cos(o_angle)
                        o_dy = 3 * math.sin(o_angle)

                        o_arrow = ax.quiver(row[x_col], row[y_col], o_dx, o_dy, angles='xy', scale_units='xy', width = 0.008, scale=1, alpha=0.5, color = 'blue')
                        o_arrows.append(o_arrow)
                        
                    except ValueError:
                        continue

    ani = FuncAnimation(fig, update, frames=range(1, n+1), repeat=True, interval=200)
    plt.close(ani._fig)

    # Display the animation in the notebook
    return HTML(ani.to_jshtml())

