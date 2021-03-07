# GNN-on-3d-points

### Abstract:
   This research focuses on 3D shape classification. Our goal is to predict the category of shapes consisting of 3D data points. We aim to implement Graph Neural Network models and compare the performances with PointNet, a popular architecture for 3d points cloud classification tasks. Not only will we compare standard metrics such as accuracy and confusion matrix, we will also explore the model's resilience of data transformation. Besides, we also tried combining PointNet with graph pooling layers.
   
   
See project website here: https://ctwayen.github.io/Graph-Neural-Network-on-3D-Points/

Docker name: ctwayen/project_docker

Docker web path: https://hub.docker.com/repository/docker/ctwayen/project_docker

### Instruction:

   If it is the first time you running this project, please download the data through python run.py all --mode download; You could also use the parameter --method to choose knn or fix-radius to construct the graph you like. Their corresponding parameters are --k and --r.The raw dataset would automatically download into the path 'data/modelnet/ModelNet40'. The points data is stored in 'data/modelnet/modelnet_points'. The consturcted graph trainning data is in 'data/modelnet/modelnet_(knn/fix_radius)(k/r){your param}'. Do not move those files. It may cause problems
   
   We support training Pointnet and GCN two models.Using --model to choose which one you want to train: GCN or pointNet.
   
   Shared paramaters are (default values are the best combination we found):
   
   --lr：Learning rate
   
   --bs: batch size
   
   --base: dataset path. If you are using default data, you do not need to specify this.Otherwise, write the graph dataset you just constructed
   
   --data: 10 or 40. Choose 10 to run 10 categories classfication and 40 for 40 categories
   
   --epoch: epoch
   
   --val_size: validation size. For example, 0.2 will have 20 % data as validation data
   
   --model_path: The path to store trained_model. Default is 'trained_models'
   
   --ouput_path: The path to store the ouput csv file
   
   Parameters that only used in GCN:
   
   --pool: Which pooling to use. SAG or ASA
   
   --ratio: The pooling ratio. For example, 0.4 will pool out 60% of data each time
   
#### Important! Training GCN will take about 7-10 minutes for one epoch. Training PointNet will take about 50s for one epoch. Please manage your time for training process.

   
### Notebooks and results:

   Besides training your own models, we also offered few trained models. You could check notebooks/Analyzing results to see our training results for different hyperparamters setting. You could also check how to use a trained_model there.
   
   You could also check the data augmentation's effects on models in the notebook/analyzing resistence file
   
   You could check how pooling layer affect result in the notebook/analyzing pooling file

Author: @Xinrui Zhan.If you find any bug, contact me through the email: ctwayen@outlook.com 
   
   
