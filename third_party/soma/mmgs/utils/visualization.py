import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pickle
import os
import point_cloud_utils as pcu
import joblib

class Visualizer:
    def __init__(self, cfg):
        self.cfg = cfg
        # self.obj_points_names=["bak_clean_object_points.ply", "clean_object_points.ply"]
        self.gs_points_names = "point_cloud.ply"
        self.mask_name = 'cln_pc_mask.pkl'
        self.cluster_type = cfg.cluster_type
        # Generate colors in a single hue (using colormap)
        # Options: 'Blues', 'Greens', 'Greys', 'Purples', 'Oranges', 'Reds'
        self.color_list = self._generate_monochrome_colors(n_colors=20, colormap='Oranges')
        #load gs
        self._initialize()

        #visual params
        self.cluster_color_list = []
        self.width = 10
        self.margin = 2

        self.k = [80, 40]
        self.base_size = 20
        # Use a blue color from the Blues colormap (matching the cluster colors)
        cmap = cm.get_cmap('Oranges')
        base_rgba = cmap(0.5)  # Use middle blue from the colormap
        self.base_color = list(base_rgba)  # RGBA format
        self.merge_threshold = 0.03  # distance threshold for merging nearby points

    def _generate_monochrome_colors(self, n_colors=20, colormap='Blues'):
        """
        Generate colors in a single hue using matplotlib colormaps.
        
        Args:
            n_colors: number of colors to generate
            colormap: name of the colormap ('Blues', 'Greens', 'Greys', 'Purples', 'Oranges', 'Reds')
        
        Returns:
            numpy array of shape (n_colors, 3) with RGB values
        """
        cmap = cm.get_cmap(colormap)
        # Use the middle to high range of the colormap to avoid very light colors
        colors = []
        for i in range(n_colors):
            # Map from 0.3 to 0.9 to avoid too light or too dark colors
            t = 0.3 + 0.6 * (i / max(n_colors - 1, 1))
            rgba = cmap(t)
            colors.append(rgba[:3])  # Take only RGB, ignore alpha
        return np.array(colors)

    #initialize the visualizer
    def _initialize(self):
        self._load_cloud_points()

        #load cluster_mask
        self._load_cluster_mask()
    
    def _load_gs_pathes(self):
        scene_name = self.cfg.scene_list[0]
        data_dir = self.cfg.data_dir
        self.scene_dir = os.path.join(data_dir, scene_name)
        # self.mask_path = os.path.join(data_dir, scene_name, self.clean_pc_mask_name )
        for obj_points_name in self.obj_points_names:
            mov_sc_pcs_path = os.path.join(data_dir, scene_name, obj_points_name )
            if os.path.exists(mov_sc_pcs_path):
                self.mov_sc_pcs_path = mov_sc_pcs_path
                break
    
    def _load_cloud_points(self):
        scene_name = self.cfg.scene_list[0]
        data_dir = self.cfg.data_dir
        self.scene_dir = os.path.join(data_dir, scene_name)
        gs_path = os.path.join(self.scene_dir, 'pi3', 'gs', 'point_cloud', 'iteration_10000', self.gs_points_names)
        # mask_path = os.path.join(self.scene_dir, self.mask_name)

        #load gs point cloud
        point_cloud = pcu.load_mesh_v(gs_path)

        # #load mask
        # with open(mask_path, 'rb') as f:
        #     mask = pickle.load(f)

        # self.point_cloud = point_cloud[mask]
        self.point_cloud = point_cloud

        self.n_points = self.point_cloud.shape[0]

    def _load_pkl(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data['p2c']

    def _load_cluster_mask(self):
        #get path
        cluster_mask_dir = os.path.join(self.scene_dir, 'cluster_mask')
        cluster_type_list = os.listdir(cluster_mask_dir)
        if self.cluster_type in cluster_type_list:
            pass
        else:
            assert len(cluster_type_list) >= 1, f"no cluster mask found in {cluster_mask_dir}"
            self.cluster_type = cluster_type_list[0]
        cluster_mask_path = os.path.join(cluster_mask_dir, self.cluster_type)
        cluster_mask_files = os.listdir(cluster_mask_path)
        # file_name = cluster_mask_files[0].replace(".pkl", ".png")
        # self.save_path = os.path.join(cluster_mask_path, file_name)

        self.cluster_mask_list = []
        self.n_cluster_layers =  []
        self.save_pathes = []
        for cluster_mask_name in cluster_mask_files:
            if cluster_mask_name.endswith('.pkl'):
                cluster_mask_path = os.path.join(cluster_mask_path, cluster_mask_name)
                cluster_mask = self._load_pkl(cluster_mask_path)
                self.save_pathes.append(cluster_mask_path.replace('.pkl', '.png'))
                self.cluster_mask_list.append(cluster_mask)
                self.n_cluster_layers.append(len(cluster_mask))

        # #load cluster mask data
        # self.cluster_mask_list = self._load_pkl(cluster_mask_path)
        # self.n_cluster_layer = len(self.cluster_mask_list)
    
    #search for cluster mask ids
    def _search_cluster_mask_ids(self, i_layer, i_particle,cluster_mask_list, n_cluster_layer):
        if i_layer + n_cluster_layer < 0:
            i_layer = n_cluster_layer - 1
        elif i_layer < 0:
            i_layer = n_cluster_layer + i_layer
        elif i_layer >= n_cluster_layer:
            i_layer = n_cluster_layer - 1
        
        for i in range(i_layer):
            cluster_mask = cluster_mask_list[i]
            max_id = np.max(cluster_mask)
            # print(f"Layer {i+1} max id: {max_id}")
            if i == 0:
                try:
                    i_cluster = cluster_mask[i_particle]
                except:
                    i_cluster = max_id+1
            else:
                try:
                    i_cluster = cluster_mask[i_cluster]
                except:
                    i_cluster = max_id + 1
        return i_cluster

    def _get_cluster_color(self, i_layer, cluster_mask_list,  n_cluster_layer):
        #1. get cluster ids
        cluster_ids = self._get_cluster_ids(i_layer, cluster_mask_list, n_cluster_layer)
        cluster_color = self.color_list[cluster_ids % len(self.color_list)]
        return cluster_color

    def _get_cluster_ids(self, i_layer, cluster_mask_list, n_cluster_layer):
        cluster_ids = []
        for i_particle in range(self.n_points):
            i_cluster = self._search_cluster_mask_ids(i_layer, i_particle, cluster_mask_list, n_cluster_layer)
            cluster_ids.append(i_cluster)
        return np.array(cluster_ids)
    
    def _transfer(self, point_cloud, flips):
        ret = np.zeros_like(point_cloud)
        for ids, f in enumerate(flips):
            if f == 1:
                mid = 0.5 * ( point_cloud[:, ids].min() + point_cloud[:, ids].max() )
                ret[:, ids] = mid - (point_cloud[:, ids] - mid)
            else:
                ret[:, ids] = point_cloud[:, ids]
        return ret

    def _farthest_points(self, points, m=5):
        """Farthest point sampling to spread representatives."""
        if len(points) == 0:
            return np.zeros((0, 3))
        m = min(m, len(points))
        # start with the point farthest from centroid for determinism
        centroid = points.mean(axis=0)
        idx0 = np.argmax(np.linalg.norm(points - centroid, axis=1))
        chosen = [idx0]
        dists = np.linalg.norm(points - points[idx0], axis=1)
        for _ in range(1, m):
            idx = np.argmax(dists)
            chosen.append(idx)
            dists = np.minimum(dists, np.linalg.norm(points - points[idx], axis=1))
        reps = points[chosen]
        if len(reps) < m:  # pad if needed
            reps = np.vstack([reps, np.tile(reps[-1], (m - len(reps), 1))])
        return reps

    def _merge_nearby_points(self, centroids, sizes, colors, threshold):
        """Merge points that are closer than threshold distance."""
        if len(centroids) == 0:
            return centroids, sizes, colors
        
        # Use a greedy approach: iterate through points and merge nearby ones
        merged_centroids = []
        merged_sizes = []
        merged_colors = []
        used = np.zeros(len(centroids), dtype=bool)
        
        for i in range(len(centroids)):
            if used[i]:
                continue
            
            # Find all points within threshold distance
            distances = np.linalg.norm(centroids - centroids[i], axis=1)
            nearby_mask = (distances < threshold) & (~used)
            
            # Merge these points: weighted average by size
            nearby_centroids = centroids[nearby_mask]
            nearby_sizes = sizes[nearby_mask]
            nearby_colors = colors[nearby_mask]
            
            # Weighted centroid
            total_size = np.sum(nearby_sizes)
            merged_centroid = np.sum(nearby_centroids * nearby_sizes[:, np.newaxis], axis=0) / total_size
            
            # Use the color of the first (representative) point
            merged_color = colors[i]
            
            merged_centroids.append(merged_centroid)
            merged_sizes.append(total_size)
            merged_colors.append(merged_color)
            
            # Mark these points as used
            used[nearby_mask] = True
        
        return np.array(merged_centroids), np.array(merged_sizes), np.array(merged_colors)

    def _compute_layer_data(self, i, point_cloud, cluster_mask_list, n_cluster_layer):
        """Compute centroids, sizes for a given layer i."""
        # cluster ids and colors
        cluster_ids = self._get_cluster_ids(i, cluster_mask_list, n_cluster_layer)
        unique_ids, counts = np.unique(cluster_ids, return_counts=True)
        cluster_color = self.color_list[unique_ids % len(self.color_list)]
        if len(self.cluster_color_list) < i:
            self.cluster_color_list.append(cluster_color)
        else:
            cluster_color = self.cluster_color_list[i-1]

        # aggregate: one marker per cluster, size ~ cluster population
        centroids = []
        sizes = []
        colors = []
        k = self.k[n_cluster_layer-i-1]
        for idx, (cid, cnt) in enumerate(zip(unique_ids, counts)):
            mask = cluster_ids == cid
            cluster_points = point_cloud[mask]
            centroid = cluster_points.mean(axis=0)
            # marker area ~ count and cluster spatial spread; keep readable
            spread = np.linalg.norm(cluster_points.max(axis=0) - cluster_points.min(axis=0)) if len(cluster_points) > 0 else 0
            size = np.clip(np.sqrt(cnt) * k * (0.5 + 0.5 * spread), 30, 5e6)
            color = cluster_color[idx]

            # deepest layer: show 5 markers per cluster; otherwise 1 marker
            if i == n_cluster_layer - 1:
                reps = self._farthest_points(cluster_points, m=5)
                if len(reps) < 5:  # pad if empty
                    reps = np.vstack([reps, np.tile(centroid, (5 - len(reps), 1))])
                centroids.extend(reps)
                sizes.extend([size] * 5)
                colors.extend([color] * 5)
            else:
                centroids.append(centroid)
                sizes.append(size)
                colors.append(color)
        centroids = np.array(centroids)
        sizes = np.array(sizes)
        colors = np.array(colors)
        return centroids, sizes, colors

    def _clean_axis(self, ax):
        """Remove axis, ticks, grid, and background from 3D plot."""
        # Remove grid
        ax.grid(False)
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # Remove labels
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        # Make panes transparent
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        # Remove pane edges
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        # Remove axis lines (the black lines)
        ax.xaxis.line.set_linewidth(0)
        ax.yaxis.line.set_linewidth(0)
        ax.zaxis.line.set_linewidth(0)

    def _visualization_hierachy(self,  save_path, cluster_mask_list, n_cluster_layer, flips=[0, 0, 0]):
        hight = n_cluster_layer * self.width + (n_cluster_layer - 1) * self.margin
        fig = plt.figure(figsize=(hight, self.width))
        
        # get point cloud with flips applied
        point_cloud = self._transfer(self.point_cloud, flips=flips)
        
        # Pre-compute layer 2 (i=1) data BEFORE merging for layer 1 to use
        layer2_centroids_before_merge, layer2_sizes_before_merge, _ = self._compute_layer_data(1, point_cloud, cluster_mask_list, n_cluster_layer)
        
        for i in range(0, n_cluster_layer):
            # Special handling for layer 0: copy layer 2's positions and sizes (before merge), but use uniform color
            if i == 0:
                ax = fig.add_subplot(1, n_cluster_layer, i+1, projection='3d')
                # Use layer 2's centroids and sizes BEFORE merging, but uniform color
                centroids = layer2_centroids_before_merge
                sizes = layer2_sizes_before_merge
                colors = np.tile(self.base_color, (len(centroids), 1))
                ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c=colors, s=sizes)
                ax.set_title(f'Layer {i+1}')
                self._clean_axis(ax)
                continue

            # For other layers, compute and plot normally
            centroids, sizes, colors = self._compute_layer_data(i, point_cloud, cluster_mask_list, n_cluster_layer)
            
            # Special handling for layer 2 (i=1): apply merging
            if i == 1:
                centroids, sizes, colors = self._merge_nearby_points(centroids, sizes, colors, self.merge_threshold)

            #plot aggregated clusters
            ax = fig.add_subplot(1, n_cluster_layer, i+1, projection='3d')
            ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c=colors, s=sizes)
            ax.set_title(f'Layer {i+1}')
            self._clean_axis(ax)

        plt.tight_layout()
        print(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    
    def run(self):
        FLIPS = [
            [0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]
        ]
        for flip in FLIPS:
            for save_path, clean_point_cloud, n_cluster_layer in zip(self.save_pathes, self.cluster_mask_list, self.n_cluster_layers):
                flip_name = '_'.join([str(f) for f in flip])
                save_path = save_path.replace('.png', f'_{flip_name}.png')
                self._visualization_hierachy(save_path, clean_point_cloud, n_cluster_layer, flips=flip)