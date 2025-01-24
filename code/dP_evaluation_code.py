import numpy as np

error = 0
well_index = 0
num_global = 100*100*5*24
num_well_global = 10*10*5*24
number_well_LGR = (40*40-20*20)*25*24 + (40*40-20*20)*50*24 + (40*40-8*8)*50*24 + 40*40*50*24

num_wells = np.loadtxt('../datasets/num_list_test.txt').astype(np.int8)
ptmax0 = np.load('../datasets/dP4_all_reservoirs_pt_max.npy').astype(np.float32).reshape((301, 24, 1, 1, 1))
ptmax = np.load('../datasets/dP4_all_wells_pt_max.npy').astype(np.float32).reshape((739, 24, 1, 1, 1))
IJ=np.load('../datasets/IJ_global.npy').astype(np.int8)

path = './'

# print(np.load(f'{path}/dP_GLOBAL_test_output.npz')['output'].shape)
# print(np.load(f'{path}dP_LGR1_test_output.npz')['output'].shape)
# print(np.load(f'{path}/dP_LGR2_test_output.npz')['output'].shape)
# print(np.load(f'{path}/dP_LGR3_test_output.npz')['output'].shape)
# print(np.load(f'{path}/dP_LGR4_test_output.npz')['output'].shape)

# print(np.load(f'{path}FNO_dP_LGR/pred_dP_GLOBAL_seq.npy').shape)
# print(np.load(f'{path}FNO_dP_LGR/pred_dP_LGR1_seq.npy').shape)
# print(np.load(f'{path}FNO_dP_LGR/pred_dP_LGR2_seq.npy').shape)
# print(np.load(f'{path}FNO_dP_LGR/pred_dP_LGR3_seq.npy').shape)
# print(np.load(f'{path}FNO_dP_LGR/pred_dP_LGR4_seq.npy').shape)

gt0=np.moveaxis(np.squeeze(np.load(f'{path}/dP_GLOBAL_test_output.npz')['output']), 4, 1)/ptmax0
gt1=np.moveaxis(np.squeeze(np.load(f'{path}/dP_LGR1_test_output.npz')['output']), 4, 1)/ptmax
gt2=np.moveaxis(np.squeeze(np.load(f'{path}/dP_LGR2_test_output.npz')['output']), 4, 1)/ptmax
gt3=np.moveaxis(np.squeeze(np.load(f'{path}/dP_LGR3_test_output.npz')['output']), 4, 1)/ptmax
gt4=np.moveaxis(np.squeeze(np.load(f'{path}/dP_LGR4_test_output.npz')['output']), 4, 1)/ptmax

pred0=np.squeeze(np.load(f'{path}/dP0_output.npz')['arr_0'])/ptmax0
pred1=np.squeeze(np.load(f'{path}/dP1_output.npz')['arr_0'])/ptmax
pred2=np.squeeze(np.load(f'{path}/dP2_output.npz')['arr_0'])/ptmax
pred3=np.squeeze(np.load(f'{path}/dP3_output.npz')['arr_0'])/ptmax
pred4=np.squeeze(np.load(f'{path}/dP4_output.npz')['arr_0'])/ptmax

pred1[:,:,19-1:38,11-1:30] = gt1[:,:,19-1:38,11-1:30]
pred2[:,:,11-1:30,11-1:30] = gt2[:,:,11-1:30,11-1:30]
pred3[:,:,17-1:24,17-1:24] = gt3[:,:,17-1:24,17-1:24]

error_array = []
for i in range(301):
    num_wells_i = num_wells[i]
    error_i=0
    error0_i=np.abs(pred0[i]-gt0[i])
    for j in range(0, num_wells_i):
        IJ_well = IJ[well_index]
        error0_i[:, IJ_well[0]-1:IJ_well[1], IJ_well[2]-1:IJ_well[3], :] = 0
        error1_i = np.sum(np.abs(pred1[well_index] - gt1[well_index]))
        error2_i = np.sum(np.abs(pred2[well_index] - gt2[well_index]))
        error3_i = np.sum(np.abs(pred3[well_index] - gt3[well_index]))
        error4_i = np.sum(np.abs(pred4[well_index] - gt4[well_index]))
        error_i=error_i+error1_i+error2_i+error3_i+error4_i
        well_index = well_index+1
    error_i=error_i+np.sum(error0_i)
    number = (num_global-num_well_global*num_wells_i)+number_well_LGR*num_wells_i
    print(number)
    error_i = error_i/number
    error = error+error_i
    print(f'{i}: {error_i}')
    error_array.append(error_i)

np.save('each_sample_error_301.npy', error_array)

error=error/301
print(f'average error: {error}')


## PER TIME STEP

# error_array = []

# num_global = 100*100*5*1
# num_well_global = 10*10*5*1
# number_well_LGR = (40*40-20*20)*25*1 + (40*40-20*20)*50*1 + (40*40-8*8)*50*1 + 40*40*50*1

# for t in range(24):
#     pred0_t = pred0[:, t:t+1, :, :, :]
#     gt0_t = gt0[:, t:t+1, :, :, :]
#     pred1_t = pred1[:, t:t+1, :, :, :]
#     gt1_t = gt1[:, t:t+1, :, :, :]
#     pred2_t = pred2[:, t:t+1, :, :, :]
#     gt2_t = gt2[:, t:t+1, :, :, :]
#     pred3_t = pred3[:, t:t+1, :, :, :]
#     gt3_t = gt3[:, t:t+1, :, :, :]
#     pred4_t = pred4[:, t:t+1, :, :, :]
#     gt4_t = gt4[:, t:t+1, :, :, :]
#     well_index = 0
#      = []

#     for i in range(301):
#         num_wells_i = num_wells[i]
#         error_i=0
#         error0_i=np.abs(pred0_t[i]-gt0_t[i])
#         for j in range(0, num_wells_i):
#             IJ_well = IJ[well_index]
#             error0_i[:, IJ_well[0]-1:IJ_well[1], IJ_well[2]-1:IJ_well[3], :] = 0
#             error1_i = np.sum(np.abs(pred1_t[well_index] - gt1_t[well_index]))
#             error2_i = np.sum(np.abs(pred2_t[well_index] - gt2_t[well_index]))
#             error3_i = np.sum(np.abs(pred3_t[well_index] - gt3_t[well_index]))
#             error4_i = np.sum(np.abs(pred4_t[well_index] - gt4_t[well_index]))
#             error_i=error_i+error1_i+error2_i+error3_i+error4_i
#             well_index = well_index+1
#         error_i=error_i+np.sum(error0_i)
#         number = (num_global-num_well_global*num_wells_i)+number_well_LGR*num_wells_i
#         error_i = error_i/number
#         error = error+error_i
#         # print(f'{i}: {error_i}')
#         .append(error_i)
#     error_array.append()

# print(np.array(error_array).shape)
# print(np.mean(np.array(error_array), axis=1))