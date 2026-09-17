'''
from matlab_ports.plot_mean_trace import plot_mean_trace

plot_mean_trace(
    'EMXJ21_0406',
    13,
    [[20, 30, 40], [20, 30, 40]],
    'data'
)
'''
from matlab_ports.play_data_trial import play_data_trial

play_data_trial(
    video1='data/EMXJ21_20220406_Trial013_bottomView_CamImgSync.mp4',
    imaging='data/EMXJ21_0406-013.mat',
    clim_abs=1.0
)