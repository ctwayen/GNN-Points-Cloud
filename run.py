from src import graph_construct, visual
import h5py
import networkx as nx
import json
import argparse
def main():
    parser = argparse.ArgumentParser(description='GNN-Points-Cloud')
    parser.add_argument('test', action = 'store_true', help='running test')
    parser.add_argument('--mode', type=str, default='eda', choices=['eda', 'train', 'predict'],
                        help='Evaluate, train, or predict')
    parser.add_argument('--model', type=str, default='pointNet', choices=['pointNet', 'GCN', 'GCN_PointNet'],
                        help='Model to use')
    parser.add_argument('--graph_image_path', type=str, default=None,
                        help='The path to store the output graph_visual_image_path')
    parser.add_argument('--points_image_path', type=str, default=None,
                        help='The path to store the output points_visual_image_path')
    parser.add_argument('--visual_base', type=str, default='x', choices=['x', 'y', 'z'],
                        help='The base axis to visual graph and points')
    args = parser.parse_args()
    if args.test:
        print("Running test will read a test points cloud data, consturct graph based on it, and then visualize it.")
        print("In the later implementation, we will also predict the class of the test sample")
        print("You could specify the path for the generated image with argument graph_image_path and points_image_path")
        print("If no path speficied, program will automatically store them as 1.png and 2.png in the cuurent directory")
        f = h5py.File('data/test.h5', 'r')
        test = f['points'][:]
        test_1000 = graph_construct.pts_sample(test, 1000)
        test_1000 = graph_construct.pts_norm(test_1000)
        A = graph_construct.graph_construct_radius(test_1000, 0.1)
        G = visual.graph(A)
        visual.draw_graph(G, test_1000, path='1.png')
        visual.visual(test_1000, path='2.png')
        f.close()
    else:
        print('Still work in progress')
if __name__ == '__main__':
    main()