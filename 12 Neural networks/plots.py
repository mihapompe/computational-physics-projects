import numpy as np
import matplotlib.pyplot as plt
import plotting
import pandas as pd

epochs = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,500]
batch = [50, 100, 500, 1000, 5000, 10000]
labels = ['lepton-pT', 'lepton-eta', 'missing-energy', 'jet_1-pt', 'jet_1-eta',
 'jet_2-pt', 'jet_2-eta', 'jet_3-pt', 'jet_3-eta', 'jet_4-pt', 'jet_4-eta',
 'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb']

# =================================================
# data = []
# for e in epochs:
#     data.append(np.load(f"results/epochs_{e}.npy", allow_pickle=True))

# eval, eval_training, dt, epochs, BATCH_SIZE, num_layers, num_nodes, [y_val, y_score], auc

# plt.plot(epochs, [data[i][0][1] for i in range(len(data))], label="Accuracy")
# plt.plot(epochs, [data[i][1][1] for i in range(len(data))], label="Training accuracy")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.title("Model accuracy vs number of epochs")
# plt.legend()
# plt.savefig("graphs/epochs_accuracy.pdf")
# plt.clf()

# plt.plot(epochs, [data[i][0][0] for i in range(len(data))], label="Loss")
# plt.plot(epochs, [data[i][1][0] for i in range(len(data))], label="Training loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("Model loss vs number of epochs")
# plt.legend()
# plt.savefig("graphs/epochs_loss.pdf")
# plt.clf()

# plt.plot(epochs, [data[i][-1] for i in range(len(data))], label="AUC")
# plt.xlabel("Epochs")
# plt.ylabel("AUC")
# plt.title("AUC vs number of epochs")
# plt.legend()
# plt.savefig("graphs/epochs_auc.pdf")
# plt.clf()

# plotting.plot_roc(data[-3][7][0],data[-3][7][1])
# plotting.plot_score(data[-3][7][0],data[-3][7][1])
# =================================================
data = []
for b in batch:
    data.append(np.load(f"results/batch_{b}.npy", allow_pickle=True))

# plt.plot(batch, [data[i][0][1] for i in range(len(data))], label="Accuracy")
# plt.plot(batch, [data[i][1][1] for i in range(len(data))], label="Training accuracy")
# plt.xlabel("Batch size")
# plt.ylabel("Accuracy")
# plt.title("Model accuracy vs batch size")
# plt.legend()
# plt.savefig("graphs/batch_accuracy.pdf")
# plt.clf()

# plt.plot(batch, [data[i][0][0] for i in range(len(data))], label="Loss")
# plt.plot(batch, [data[i][1][0] for i in range(len(data))], label="Training loss")
# plt.xlabel("Batch size")
# plt.ylabel("Loss")
# plt.title("Model loss vs batch size")
# plt.legend()
# plt.savefig("graphs/batch_loss.pdf")
# plt.clf()

t = [data[i][-1] for i in range(len(data))]
mx = max(t)
plt.plot(batch, t, label="AUC")
plt.xlabel("Batch size")
plt.ylabel("AUC")
plt.title(f"AUC vs batch size\nMax at ({batch[t.index(mx)]},{round(mx,2)})")
plt.legend()
plt.savefig("graphs/batch_auc.pdf")
plt.clf()

# =================================================
# data = []
# layout_x = []
# layout_y = []
# for i, n_layers in enumerate(range(1, 10)):
#     data.append([])
#     layout_x.append([])
#     layout_y.append([])
#     for j, n_nodes in enumerate(range(10, 100, 10)):
#         data[i].append(np.load(f"results/layers_nodes_{n_layers}_{n_nodes}.npy", allow_pickle=True))
#         layout_x[i].append(n_layers)
#         layout_y[i].append(n_nodes)

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# layout_x = np.array(layout_x)
# layout_y = np.array(layout_y)
# acc = np.array([[data[i][j][0][1] for j in range(len(layout_x[0]))] for i in range(len(layout_x))])
# ax.plot_surface(layout_x, layout_y, acc, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
# ax.set_title('Accuracy for different topologies of NN')
# ax.set_xlabel("Number of layers")
# ax.set_ylabel("Number of nodes")
# plt.savefig("graphs/layers_nodes_accuracy.pdf")
# plt.clf()

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# layout_x = np.array(layout_x)
# layout_y = np.array(layout_y)
# acc = np.array([[data[i][j][0][0] for j in range(len(layout_x[0]))] for i in range(len(layout_x))])
# ax.plot_surface(layout_x, layout_y, acc, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
# ax.set_title('Loss for different topologies of NN')
# ax.set_xlabel("Number of layers")
# ax.set_ylabel("Number of nodes")
# plt.savefig("graphs/layers_nodes_loss.pdf")
# plt.clf()

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# layout_x = np.array(layout_x)
# layout_y = np.array(layout_y)
# acc = np.array([[data[i][j][-1] for j in range(len(layout_x[0]))] for i in range(len(layout_x))])
# ax.plot_surface(layout_x, layout_y, acc, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
# ax.set_title('AUC for different topologies of NN')
# ax.set_xlabel("Number of layers")
# ax.set_ylabel("Number of nodes")
# plt.savefig("graphs/layers_nodes_auc.pdf")
# plt.clf()

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# layout_x = np.array(layout_x)
# layout_y = np.array(layout_y)
# acc = np.array([[data[i][j][2] for j in range(len(layout_x[0]))] for i in range(len(layout_x))])
# ax.plot_surface(layout_x, layout_y, acc, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
# ax.set_title('Training time for different topologies of NN')
# ax.set_xlabel("Number of layers")
# ax.set_ylabel("Number of nodes")
# plt.savefig("graphs/layers_nodes_time.pdf")
# plt.clf()

# =================================================

# data = []
# for i in range(18):
#     data.append(np.load(f"results/remove_{i}.npy", allow_pickle=True))
# d = [data[i][-1]-0.6 for i in range(18)]

# a = np.array([labels, d]).T
# df = pd.DataFrame(a, columns = ["labels", "data"])
# df = df.sort_values("data")
# df["labels"][2] = "miss_en"
# plt.bar(df["labels"], df["data"].astype(float))
# plt.xticks(rotation=60)
# plt.ylabel("AUC - 0.6")
# plt.subplots_adjust(bottom=0.2)
# plt.title("AUC vs excluded variable")
# plt.savefig("graphs/excluded_variable_auc.pdf")
# plt.show()
# =================================================

# data = []
# num_est = [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,50,75,100,150,200]
# for n_est in num_est:
#     data.append(np.load(f"results/bdt_{n_est}.npy", allow_pickle=True))
# data = np.array(data)
# auc = data[:,-1]

# plt.plot(num_est, auc)
# plt.title("AUC vs number of estimatiors")
# plt.xlabel("Number of estimators")
# plt.ylabel("AUC")
# plt.savefig("graphs/bdt_auc_estimators.pdf")
# plt.clf()

# plt.plot(num_est, data[:,0], label="score")
# plt.plot(num_est, data[:,1], label="training score")
# plt.title("Score vs number of estimatiors")
# plt.xlabel("Number of estimators")
# plt.ylabel("Score")
# plt.legend()
# plt.savefig("graphs/bdt_score_estimators.pdf")
# plt.show()

# =================================================
# Optimal DNN

# data  = np.load(f"results/optimal_dnn_0.npy", allow_pickle=True)
# print(data[-1])
# plotting.plot_roc(data[7][0],data[7][1])
# plotting.plot_score(data[7][0],data [7][1])
