import numpy as np
import matplotlib.pyplot as plt
import random
import os
import time
# noise_epistemic = np.random.randn(10)

def generate_bg():
    pid = os.getpid()
    temp = int(time.time()) + pid
    np.random.seed(temp)
    runway_wid_list = [45,60]
    time_step = 20
    # runway_width = random.choice(runway_wid_list)
    g_width = np.random.randint(700,900)
    g_height = np.random.randint(500,700)
    print('ground size',g_width,g_height)
    flight_left_size = np.random.randint(35,60)
    
    flight_right_size = np.random.randint(35,60)
    # print(flight_left_size,flight_left_size/2)
    
    junction_center = np.array([np.random.randint(200,g_width-200),np.random.randint(100,g_height-100)])
    
    for runway_width in runway_wid_list:
        if runway_width > max([flight_left_size,flight_right_size]):
            break
    
    noise_epistemic = 10 * np.random.randn(3)
    b_g = {}
    b_g['T_1'] = np.random.randint(10,15)*time_step
    b_g['T_2'] = b_g['T_1'] + np.random.randint(5,10)*time_step
    b_g['d_safe'] = (flight_left_size+flight_right_size)/2
    b_g['edge_1'] = np.array([[0,0] + noise_epistemic[0],[0,g_height] + noise_epistemic[1]])
    b_g['edge_2'] = np.array([[0,g_width] + noise_epistemic[0],[g_height,g_height] + noise_epistemic[1]])
    b_g['edge_3'] = np.array([[g_width,g_width] + noise_epistemic[0],[g_height,0] + noise_epistemic[1]])
    b_g['edge_4'] = np.array([[g_width,0] + noise_epistemic[0],[0,0] + noise_epistemic[1]])
    b_g['track_hori_bot'] = np.array([[0,g_width] + noise_epistemic[0],[junction_center[1]-runway_width/2,junction_center[1]-runway_width/2] + noise_epistemic[1]])
    b_g['track_hori_top_left'] = np.array([[0,junction_center[0]-runway_width/2] + noise_epistemic[0],[junction_center[1]+runway_width/2,junction_center[1]+runway_width/2] + noise_epistemic[1]])
    b_g['track_hori_top_right'] = np.array([[junction_center[0]+runway_width/2,g_width] + noise_epistemic[0],[junction_center[1]+runway_width/2,junction_center[1]+runway_width/2] + noise_epistemic[1]])
    b_g['track_vert_left'] = np.array([[junction_center[0]-runway_width/2,junction_center[0]-runway_width/2] + noise_epistemic[0],[junction_center[1]+runway_width/2,g_height] + noise_epistemic[1]])
    b_g['track_vert_right'] = np.array([[junction_center[0]+runway_width/2,junction_center[0]+runway_width/2] + noise_epistemic[0],[junction_center[1]+runway_width/2,g_height] + noise_epistemic[1]])
    b_g['goal'] = np.array([[junction_center[0]-runway_width/2,junction_center[0]+runway_width/2] + noise_epistemic[0],[g_height-50,g_height-50] + noise_epistemic[1]])
    # b_g['runway'] = (junction_center[0]-runway_width/2+ noise_epistemic[0],junction_center[0]+runway_width/2+ noise_epistemic[0],junction_center[1]+runway_width/2+ noise_epistemic[1],junction_center[1]+runway_width/2+ noise_epistemic[1])
    b_g['runway'] = np.array([[junction_center[0]-runway_width/2,junction_center[0]+runway_width/2] + noise_epistemic[0],[junction_center[1]+runway_width/2,junction_center[1]+runway_width/2] + noise_epistemic[1]])
    flight_left_x = np.random.randint(0 + flight_left_size/2, junction_center[0]- runway_width/2 -flight_left_size/2)
    flight_left_y = np.random.randint(junction_center[1]-runway_width/2 +flight_left_size/2,junction_center[1]+runway_width/2-flight_left_size/2)
    
    flight_right_x = np.random.randint(junction_center[0] + runway_width/2 +flight_right_size/2,g_width - flight_right_size/2)
    flight_right_y = np.random.randint(junction_center[1]-runway_width/2+flight_right_size/2,junction_center[1]+runway_width/2-flight_right_size/2)
    if noise_epistemic[-1] >= 0:
        b_g['flight_1'] = np.array([[flight_left_x]+ noise_epistemic[0],[flight_left_y]+ noise_epistemic[1]])
        b_g['flight_size1'] = flight_left_size
        b_g['flight_2'] = np.array([[flight_right_x]+ noise_epistemic[0],[flight_right_y]+ noise_epistemic[1]])
        b_g['flight_size2'] = flight_right_size
    else:
        b_g['flight_2'] = np.array([[flight_left_x]+ noise_epistemic[0],[flight_left_y]+ noise_epistemic[1]])
        b_g['flight_size2'] = flight_left_size
        b_g['flight_1'] = np.array([[flight_right_x]+ noise_epistemic[0],[flight_right_y]+ noise_epistemic[1]])
        b_g['flight_size1'] = flight_right_size
    # b_g['flight_len'] = 80
    # b_g['flight_wid'] = 80
    # print(b_g['flight_1'])
    b_g['obstacle_1'] = (b_g['edge_1'][0,0],b_g['track_vert_left'][0,0], b_g['track_vert_left'][1,0],b_g['edge_1'][1,1])
    b_g['obstacle_2'] = (b_g['track_vert_right'][0,0],b_g['edge_3'][0,0],b_g['track_vert_right'][1,0],b_g['edge_3'][1,0])
    b_g['obstacle_3'] = (b_g['edge_1'][0,0],b_g['edge_3'][0,0],b_g['edge_4'][1,1],b_g['track_hori_bot'][1,0])
    
    b_g['background'] = (0+noise_epistemic[0], g_width+noise_epistemic[0], 0+noise_epistemic[1], g_height+noise_epistemic[1])
    
    b_g['runway_tuple'] = (b_g['runway'][0,0],b_g['goal'][0,1],b_g['runway'][1,0],b_g['goal'][1,1])
    
    b_g['goal_tuple'] = (b_g['track_vert_left'][0,0],b_g['track_vert_right'][0,0],b_g['goal'][1,1],b_g['edge_2'][1,1])
    
    b_g['track'] = (b_g['track_hori_bot'][0,0],b_g['track_hori_bot'][0,1],b_g['track_hori_bot'][1,0],b_g['track_hori_top_right'][1,1])
    
    return b_g

def plot_bg(b_g):
    # print(background['edge_1'].shape)
    
    plt.figure(figsize=(16,10))
    # plt.xlim(-20,720)
    # plt.xticks([])
    # plt.ylim(-20,520)
    # plt.yticks([])
    plt.plot(b_g['edge_1'][0],b_g['edge_1'][1],color='k')
    plt.plot(b_g['edge_2'][0],b_g['edge_2'][1],color='k')
    plt.plot(b_g['edge_3'][0],b_g['edge_3'][1],color='k')
    plt.plot(b_g['edge_4'][0],b_g['edge_4'][1],color='k')
    plt.plot(b_g['track_hori_bot'][0],b_g['track_hori_bot'][1],color='r')
    plt.plot(b_g['track_hori_top_left'][0],b_g['track_hori_top_left'][1],color='r')
    plt.plot(b_g['track_hori_top_right'][0],b_g['track_hori_top_right'][1],color='r')
    plt.plot(b_g['track_vert_left'][0],b_g['track_vert_left'][1],color='r')
    plt.plot(b_g['track_vert_right'][0],b_g['track_vert_right'][1],color='r')
    plt.plot(b_g['goal'][0],b_g['goal'][1],color='g')
    plt.fill_between(x=[b_g['flight_1'][0,0]+0.5*b_g['flight_size1'],b_g['flight_1'][0,0]-0.5*b_g['flight_size1']],y1=b_g['flight_1'][1,0]-0.5*b_g['flight_size1'],y2=b_g['flight_1'][1,0]+0.5*b_g['flight_size1'],color='blue')
    plt.fill_between(x=[b_g['flight_2'][0,0]+0.5*b_g['flight_size2'],b_g['flight_2'][0,0]-0.5*b_g['flight_size2']],y1=b_g['flight_2'][1,0]-0.5*b_g['flight_size2'],y2=b_g['flight_2'][1,0]+0.5*b_g['flight_size2'],color='blue')
    plt.fill_between(x=[b_g['edge_1'][0,0],b_g['track_vert_left'][0,0]],y1=b_g['track_vert_left'][1,0],y2=b_g['edge_1'][1,1],color='gray')
    plt.fill_between(x=[b_g['track_vert_right'][0,0],b_g['edge_3'][0,0]],y1=b_g['edge_3'][1,0],y2=b_g['track_vert_right'][1,0],color='gray')
    plt.fill_between(x=[b_g['edge_1'][0,0],b_g['edge_3'][0,0]],y1=b_g['edge_4'][1,1],y2=b_g['track_hori_bot'][1,0],color='gray')
    plt.fill_between(x=[b_g['track_vert_left'][0,0],b_g['track_vert_right'][0,0]],y1=b_g['goal'][1,1],y2=b_g['edge_2'][1,1],color='green')
    plt.fill_between(x=[b_g['runway'][0,0],b_g['goal'][0,1]],y1=b_g['runway'][1,0],y2=b_g['goal'][1,1],color='yellow')
    plt.plot()
    plt.show()

# b_g = generate_bg()
# plot_bg(b_g)





