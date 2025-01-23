import numpy as np

sum=0
error=0
path = './'
sum_array = np.zeros((739,))
error_array = np.zeros((739,))

level = '1'
gt1 = np.moveaxis(np.squeeze(np.load(f'{path}/SG_LGR{level}_test_output.npz')['output']), 4, 1)
print(f'gt shape: {gt1.shape}')
pred1 = np.squeeze(np.load(f'{path}/SG{level}_test_pred.npy'))
print(f'pred shape: {pred1.shape}')
# mask = np.ones((739, 24, 40 , 40, 25))
mask = np.where((abs(gt1-pred1)>0.01) | (gt1>0.01), 1, 0)
# mask[:,19-1:38,11-1:30] = 0
print(f'mask shape_before: {mask[:,19-1:38,11-1:30].shape}')
mask[:, :, 19-1:38,11-1:30, :] = 0
print(f'mask shape: {mask[:, :, 19-1:38,11-1:30, :].shape}')
sum = sum+np.sum(mask)
sum_array = sum_array+np.sum(mask, axis=(1,2,3,4))
# print(f'sum: {sum}')
error = error+np.sum(np.abs(gt1 - pred1)*mask)
error_array = error_array+np.sum(np.abs(gt1 - pred1)*mask, axis=(1,2,3,4))
# print(f'error: {error / sum}')

level = '2'
gt2 = np.moveaxis(np.squeeze(np.load(f'{path}/SG_LGR{level}_test_output.npz')['output']), 4, 1)
print(f'gt shape: {gt2.shape}')
pred2 = np.squeeze(np.load(f'{path}/SG{level}_test_pred.npy'))
print(f'pred shape: {pred2.shape}')
# mask = np.ones((739, 24, 40 , 40, 50))
mask = np.where((abs(gt2-pred2)>0.01) | (gt2>0.01), 1, 0)
mask[:, :, 11-1:30,11-1:30, :] = 0
print(f'mask shape: {mask[:, :, 11-1:30,11-1:30, :].shape}')
sum = sum+np.sum(mask)
sum_array = sum_array+np.sum(mask, axis=(1,2,3,4))
# print(f'sum: {sum}')
error = error+np.sum(np.abs(gt2 - pred2)*mask)
error_array = error_array+np.sum(np.abs(gt2 - pred2)*mask, axis=(1,2,3,4))
# print(f'error: {error}')

level = '3'
gt3 = np.moveaxis(np.squeeze(np.load(f'{path}/SG_LGR{level}_test_output.npz')['output']), 4, 1)
print(f'gt shape: {gt3.shape}')
pred3 = np.squeeze(np.load(f'{path}/SG{level}_test_pred.npy'))
print(f'pred shape: {pred3.shape}')
# mask = np.ones((739, 24, 40 , 40, 50))
mask = np.where((abs(gt3-pred3)>0.01) | (gt3>0.01), 1, 0)
mask[:, :, 11-1:30,11-1:30, :] = 0
print(f'mask shape: {mask[:, :, 11-1:30,11-1:30, :].shape}')
sum = sum+np.sum(mask)
sum_array = sum_array+np.sum(mask, axis=(1,2,3,4))
# print(f'sum: {sum}')
error = error+np.sum(np.abs(gt3 - pred3)*mask)
error_array = error_array+np.sum(np.abs(gt3 - pred3)*mask, axis=(1,2,3,4))
# print(f'error: {error}')

level = '4'
gt4 = np.moveaxis(np.squeeze(np.load(f'{path}/SG_LGR{level}_test_output.npz')['output']), 4, 1)
print(f'gt shape: {gt4.shape}')
pred4 = np.squeeze(np.load(f'{path}/SG{level}_test_pred.npy'))
print(f'pred shape: {pred4.shape}')
# mask = np.ones((739, 24, 40 , 40, 50))
mask = np.where((abs(gt4-pred4)>0.01) | (gt4>0.01), 1, 0)
print(f'mask shape: {mask.shape}')
sum = sum+np.sum(mask)
sum_array = sum_array+np.sum(mask, axis=(1,2,3,4))
print('sum array sum: ', np.sum(sum_array))
# print(f'sum: {sum}')
error = (error+np.sum(np.abs(gt4 - pred4)*mask)) / sum
error_array = (error_array+np.sum(np.abs(gt4 - pred4)*mask, axis=(1,2,3,4))) / sum_array
print(f'error: {error}')
print(sum_array.shape)
print(error_array.shape)
# np.save('sequential_not_tuned_whole_errors_301_w36_SG.npy', error_array)
print("Corrected Error: ", np.mean(error_array))


# ## PER TIME STEP
# error_array = []
# time_error = []

# for t in range(24):
#     pred1_t = pred1[:, t:t+1, :, :, :]
#     gt1_t = gt1[:, t:t+1, :, :, :]
#     pred2_t = pred2[:, t:t+1, :, :, :]
#     gt2_t = gt2[:, t:t+1, :, :, :]
#     pred3_t = pred3[:, t:t+1, :, :, :]
#     gt3_t = gt3[:, t:t+1, :, :, :]
#     pred4_t = pred4[:, t:t+1, :, :, :]
#     gt4_t = gt4[:, t:t+1, :, :, :]
#     well_index = 0

#     sum_array = np.zeros((739,))
#     error_array = np.zeros((739,))

#     level = '1'
#     mask = np.where((abs(gt1_t-pred1_t)>0.01) | (gt1_t>0.01), 1, 0)
#     # mask = np.ones((739, 24, 40 , 40, 25))
#     mask[:, :, 19-1:38,11-1:30, :] = 0
#     sum_array = sum_array+np.sum(mask, axis=(1,2,3,4))
#     error_array = error_array+np.sum(np.abs(gt1_t - pred1_t)*mask, axis=(1,2,3,4))

#     level = '2'
#     mask = np.where((abs(gt2_t-pred2_t)>0.01) | (gt2_t>0.01), 1, 0)
#     # mask = np.ones((739, 24, 40 , 40, 50))
#     mask[:, :, 11-1:30,11-1:30, :] = 0
#     sum_array = sum_array+np.sum(mask, axis=(1,2,3,4))
#     error_array = error_array+np.sum(np.abs(gt2_t - pred2_t)*mask, axis=(1,2,3,4))

#     level = '3'
#     mask = np.where((abs(gt3_t-pred3_t)>0.01) | (gt3_t>0.01), 1, 0)
#     # mask = np.ones((739, 24, 40 , 40, 50))
#     mask[:, :, 11-1:30,11-1:30, :] = 0
#     sum_array = sum_array+np.sum(mask, axis=(1,2,3,4))
#     error_array = error_array+np.sum(np.abs(gt3_t - pred3_t)*mask, axis=(1,2,3,4))

#     level = '4'
#     mask = np.where((abs(gt4_t-pred4_t)>0.01) | (gt4_t>0.01), 1, 0)
#     # mask = np.ones((739, 24, 40 , 40, 50))
#     sum_array = sum_array+np.sum(mask, axis=(1,2,3,4))
#     error_array = (error_array+np.sum(np.abs(gt4_t - pred4_t)*mask, axis=(1,2,3,4))) / sum_array
#     print(sum_array.shape)
#     print(error_array.shape)
#     print("Corrected Error: ", np.mean(error_array))
#     time_error.append(error_array)

# print(np.array(error_array).shape)
# print(np.array(time_error).shape)
# print(np.mean(np.array(error_array), axis=1))