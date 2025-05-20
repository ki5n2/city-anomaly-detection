#%%
'''IMPORTS'''
import glob
import torch
import folium
import kagglehub
import osmnx as ox # OSM 데이터
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import networkx as nx
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt

from pyproj import CRS
from xgboost import XGBClassifier
from folium.plugins import HeatMap
from torch_geometric.data import Data
from shapely.ops import nearest_points
from shapely.geometry import Point, Polygon, box
from torch_geometric.data import Data, DataLoader

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection

#%%
'''DATA'''
# 교통 시스템 이상 탐지(시공간 그래프 기반 접근)
path = kagglehub.dataset_download("arashnic/tdriver")

release_path = os.path.join(path, "release")
print("release 디렉토리 내용:", os.listdir(release_path))

log_dir = os.path.join(release_path, "taxi_log_2008_by_id")

all_files = glob.glob(os.path.join(log_dir, "*.txt"))

df_list = []
for file in all_files:
    df = pd.read_csv(file, names=["taxi_id", "timestamp", "longitude", "latitude"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df_list.append(df)

full_df = pd.concat(df_list, ignore_index=True)

print(full_df.shape)
print(full_df.head())

# GeoPandas로 변환
gdf = gpd.GeoDataFrame(full_df, geometry=[Point(xy) for xy in zip(full_df['longitude'], full_df['latitude'])])
gdf

#%%
# 도로 네트워크 매핑, 베이징 도로 네트워크
beijing_graph = ox.graph_from_bbox((41.0, 39.6, 117.5, 115.5), network_type='drive')
nodes, edges = ox.graph_to_gdfs(beijing_graph)

# 궤적 포인트를 가장 가까운 도로에 매핑
def map_to_nearest_edge(point, edges):
    distances = edges.distance(point)
    nearest_edge_idx = distances.idxmin()  # 인덱스 라벨
    return edges.loc[nearest_edge_idx]     # loc으로 인덱스 라벨 사용

# 각 GPS 포인트를 도로에 매핑
gdf['edge_id'] = gdf.geometry.apply(lambda point: map_to_nearest_edge(point, edges).name)
# gdf['edge_id'] = gdf.geometry.apply(lambda point: map_to_nearest_edge(point, edges).index_value)
gdf['edge_id'] = gdf.geometry.apply(lambda point: map_to_nearest_edge(point, edges).index_value)

# 시간적 리샘플링 및 정규화 
# 1시간 단위로 리샘플링
def resample_trajectory(group):
    return group.set_index('timestamp').resample('1H').count().reset_index()

hourly_counts = gdf.groupby(['edge_id', pd.Grouper(key='timestamp', freq='1H')]).size().reset_index(name='count')

# 시간대 추출
hourly_counts['hour'] = hourly_counts['timestamp'].dt.hour
hourly_counts['day_of_week'] = hourly_counts['timestamp'].dt.dayofweek

# 교통량 정규화
hourly_counts['normalized_count'] = hourly_counts.groupby('edge_id')['count'].transform(
    lambda x: (x - x.mean()) / x.std())

# 그래프 구성
road_graph = nx.Graph()

# # 노드 추가 (도로 교차점)
# for idx, node in nodes.iterrows():
#     road_graph.add_node(idx, pos=(node.geometry.x, node.geometry.y))

# # 엣지 추가 (도로 구간)
# for idx, edge in edges.iterrows():
#     road_graph.add_edge(edge.u, edge.v, 
#                       length=edge.length,
#                       edge_id=idx)

# # 2. 지역 기반 그래프
# # 베이징을 500x500 그리드로 분할
# lat_bins = np.linspace(39.6, 41.0, 50)
# lon_bins = np.linspace(115.5, 117.5, 50)

# df['lat_bin'] = pd.cut(df['latitude'], lat_bins, labels=False)
# df['lon_bin'] = pd.cut(df['longitude'], lon_bins, labels=False)
# df['grid_id'] = df['lat_bin'] * 50 + df['lon_bin']

# # 그리드 간 택시 이동을 엣지로 하는 그래프 구성
# region_graph = nx.DiGraph()

# # 지역 간 흐름 계산
# flows = df.groupby(['taxi_id']).apply(
#     lambda x: list(zip(x['grid_id'].iloc[:-1], x['grid_id'].iloc[1:]))
# ).explode().value_counts().reset_index()
# flows.columns = ['flow', 'count']

# # 그래프에 노드와 엣지 추가
# for grid_id in df['grid_id'].unique():
#     region_graph.add_node(grid_id)
    
# for _, row in flows.iterrows():
#     source, target = row['flow']
#     region_graph.add_edge(source, target, weight=row['count'])

# # 3. 시간적 그래프 (시간대별 교통 패턴)
# time_graph = nx.DiGraph()

# # 24시간을 노드로
# for hour in range(24):
#     time_graph.add_node(hour)
    
# # 시간대 사이의 평균 교통량을 엣지 가중치로
# hourly_patterns = hourly_counts.groupby(['hour'])['count'].mean().reset_index()

# for i in range(24):
#     next_hour = (i + 1) % 24
#     time_graph.add_edge(i, next_hour, 
#                       weight=hourly_patterns.loc[hourly_patterns['hour'] == i, 'count'].values[0])

# # 특성 행렬 생성
# # 도로 구간별 시간당 교통량 특성 행렬 (NxTxF)
# # N: 도로 구간 수, T: 시간 프레임 수, F: 특성 수

# # 시간 프레임 생성 (1시간 간격, 1주일)
# time_frames = pd.date_range(start='2008-02-02', end='2008-02-08 23:59:59', freq='1H')
# edge_ids = edges.index.unique()

# # 빈 특성 행렬 초기화
# n_edges = len(edge_ids)
# n_timeframes = len(time_frames)
# n_features = 3  # [교통량, 평균 속도, 시간대(0-23 정규화)]

# feature_tensor = np.zeros((n_edges, n_timeframes, n_features))

# # 특성 채우기
# edge_id_map = {eid: i for i, eid in enumerate(edge_ids)}
# time_map = {tf: i for i, tf in enumerate(time_frames)}

# for _, row in hourly_counts.iterrows():
#     edge_idx = edge_id_map.get(row['edge_id'], -1)
#     time_idx = time_map.get(row['timestamp'], -1)
    
#     if edge_idx >= 0 and time_idx >= 0:
#         feature_tensor[edge_idx, time_idx, 0] = row['count']  # 교통량
#         # 평균 속도는 별도 계산 필요
#         feature_tensor[edge_idx, time_idx, 2] = row['hour'] / 23.0  # 시간대 정규화

# # 인접행렬생성
# # 도로 네트워크 인접 행렬
# adjacency_matrix = nx.adjacency_matrix(road_graph).toarray()

# # 거리 기반 가중치 인접 행렬
# distance_matrix = np.zeros((n_edges, n_edges))

# # 도로 구간 간 연결성과 거리 계산
# for i, edge1_id in enumerate(edge_ids):
#     for j, edge2_id in enumerate(edge_ids):
#         if i == j:
#             distance_matrix[i, j] = 0
#         else:
#             # 두 도로 구간의 최단 거리 계산
#             edge1 = edges.loc[edge1_id]
#             edge2 = edges.loc[edge2_id]
            
#             # 도로 구간의 시작/끝 노드 사이 거리 계산
#             d1 = nx.shortest_path_length(road_graph, edge1.u, edge2.u, weight='length')
#             d2 = nx.shortest_path_length(road_graph, edge1.u, edge2.v, weight='length')
#             d3 = nx.shortest_path_length(road_graph, edge1.v, edge2.u, weight='length')
#             d4 = nx.shortest_path_length(road_graph, edge1.v, edge2.v, weight='length')
            
#             min_dist = min(d1, d2, d3, d4)
#             distance_matrix[i, j] = min_dist

# # 가우시안 커널로 변환
# sigma = distance_matrix.std()
# adjacency_weighted = np.exp(-np.square(distance_matrix) / (2 * sigma**2))

# # 임계값 적용 (상위 k개 연결만 유지)
# k = 5  # 각 노드의 상위 k개 이웃만 유지
# for i in range(n_edges):
#     threshold = np.sort(adjacency_weighted[i])[-k]
#     adjacency_weighted[i][adjacency_weighted[i] < threshold] = 0

# # 이상 레이블 생성
# def create_synthetic_anomalies(feature_tensor, anomaly_ratio=0.05):
#     # 복사본 생성
#     anomaly_tensor = feature_tensor.copy()
#     anomaly_labels = np.zeros((n_edges, n_timeframes))
    
#     # 이상 유형 1: 갑작스러운 교통량 급감
#     n_anomalies = int(n_edges * n_timeframes * anomaly_ratio)
    
#     for _ in range(n_anomalies):
#         edge_idx = np.random.randint(0, n_edges)
#         time_idx = np.random.randint(0, n_timeframes)
        
#         # 원래 값에서 70-90% 감소
#         reduction = np.random.uniform(0.7, 0.9)
#         anomaly_tensor[edge_idx, time_idx, 0] *= (1 - reduction)
#         anomaly_labels[edge_idx, time_idx] = 1
    
#     # 이상 유형 2: 지속적 교통 혼잡
#     n_congestion = int(n_anomalies * 0.3)
    
#     for _ in range(n_congestion):
#         edge_idx = np.random.randint(0, n_edges)
#         start_time = np.random.randint(0, n_timeframes - 4)  # 최소 4시간 지속
#         duration = np.random.randint(2, 5)  # 2-4시간
        
#         # 교통량 증가 (150-300%)
#         increase = np.random.uniform(1.5, 3.0)
#         for t in range(start_time, min(start_time + duration, n_timeframes)):
#             anomaly_tensor[edge_idx, t, 0] *= increase
#             anomaly_labels[edge_idx, t] = 1
    
#     return anomaly_tensor, anomaly_labels

# # 2. 통계 기반 이상 탐지로 레이블 생성
# def create_statistical_anomalies(feature_tensor, z_threshold=3.0):
#     anomaly_labels = np.zeros((n_edges, n_timeframes))
    
#     # 각 도로 구간별로 시간에 따른 교통량의 Z-score 계산
#     for edge_idx in range(n_edges):
#         edge_traffic = feature_tensor[edge_idx, :, 0]
#         if edge_traffic.std() > 0:  # 분산이 0이 아닌 경우에만
#             z_scores = (edge_traffic - edge_traffic.mean()) / edge_traffic.std()
#             # Z-score가 임계값을 넘으면 이상으로 표시
#             anomaly_labels[edge_idx] = np.abs(z_scores) > z_threshold
    
#     return anomaly_labels

# # 두 방법 결합
# synthetic_tensor, synthetic_labels = create_synthetic_anomalies(feature_tensor)
# statistical_labels = create_statistical_anomalies(feature_tensor)

# # 최종 이상 레이블
# anomaly_labels = np.logical_or(synthetic_labels, statistical_labels).astype(int)

# # 데이터 분할 및 
# # 데이터 시간 순서대로 분할 (7일 데이터)
# # 5일 훈련, 1일 검증, 1일 테스트
# train_end = int(n_timeframes * 0.7)  # 70%
# val_end = int(n_timeframes * 0.85)   # 15%

# # PyTorch Geometric 데이터 형식으로 변환
# def create_torch_geometric_data(features, adjacency, labels, time_idx):
#     # 엣지 인덱스 생성 (희소 인접 행렬)
#     edge_index = adjacency.nonzero()
#     edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
#     # 엣지 가중치
#     edge_weight = torch.tensor([adjacency[i, j] for i, j in zip(*edge_index)], dtype=torch.float)
    
#     # 노드 특성 (시간별 특성)
#     x = torch.tensor(features[:, time_idx, :], dtype=torch.float)
    
#     # 이상 레이블
#     y = torch.tensor(labels[:, time_idx], dtype=torch.long)
    
#     # 데이터 객체 생성
#     data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
#     return data

# # 시간 윈도우 슬라이딩하며 시공간 데이터 생성
# window_size = 12  # 12시간 윈도우
# pred_horizon = 1  # 1시간 후 예측

# train_dataset = []
# val_dataset = []
# test_dataset = []

# for t in range(window_size, n_timeframes - pred_horizon):
#     time_window = list(range(t - window_size, t))
#     features_window = feature_tensor[:, time_window, :]
    
#     # 3D 특성 텐서를 2D로 재구성 (각 노드가 window_size * n_features 차원 특성 가짐)
#     reshaped_features = features_window.reshape(n_edges, -1)
    
#     data = Data(
#         x=torch.tensor(reshaped_features, dtype=torch.float),
#         edge_index=torch.tensor(adjacency_weighted.nonzero(), dtype=torch.long).t().contiguous(),
#         edge_attr=torch.tensor([adjacency_weighted[i, j] for i, j in zip(*adjacency_weighted.nonzero())], dtype=torch.float),
#         y=torch.tensor(anomaly_labels[:, t], dtype=torch.long)
#     )
    
#     # 시간 순서대로 분할
#     if t < train_end:
#         train_dataset.append(data)
#     elif t < val_end:
#         val_dataset.append(data)
#     else:
#         test_dataset.append(data)

# # 데이터로더 생성
# batch_size = 32
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size)
# test_loader = DataLoader(test_dataset, batch_size=batch_size)


#%%
'''Explore and visualize'''
# 교통량 시간 패턴 시각화
hourly_avg = hourly_counts.groupby('hour')['count'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(x='hour', y='count', data=hourly_avg)
plt.title('Average Hourly Traffic Volume')
plt.xlabel('Hour of Day')
plt.ylabel('Average Traffic Count')
plt.xticks(range(0, 24, 2))
plt.grid(True, alpha=0.3)
plt.show()

# 지도 시각화 (특정 시간대 교통량 히트맵)
m = folium.Map(location=[39.9, 116.4], zoom_start=11)  # 베이징 중심

# 특정 시간대 데이터 필터링 (예: 오전 8시)
rush_hour = hourly_counts[hourly_counts['hour'] == 8]

# 히트맵 데이터 준비
heat_data = []
for _, row in rush_hour.iterrows():
    edge_id = row['edge_id']
    edge = edges.loc[edge_id]
    # 도로 중간점 좌표 가져오기
    mid_point = edge.geometry.interpolate(0.5, normalized=True)
    heat_data.append([mid_point.y, mid_point.x, row['count']])

# 히트맵 추가
HeatMap(heat_data).add_to(m)
m.save('beijing_traffic_heatmap.html')

# 그래프 시각화
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(road_graph)
nx.draw(road_graph, pos, node_size=10, alpha=0.6)
plt.title('Road Network Graph')
plt.savefig('road_network.png')
plt.close()


# %%
'''MODELING'''
class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(STGCNBlock, self).__init__()
        self.temporal_conv1 = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.graph_conv = GraphConvolution(out_channels, out_channels)
        self.temporal_conv2 = nn.Conv1d(out_channels, out_channels, kernel_size)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x, A):
        # x shape: [batch_size, num_nodes, in_channels, time_steps]
        residual = x
        # Temporal convolution 1
        t = x.permute(0, 2, 1, 3)  # [batch_size, in_channels, num_nodes, time_steps]
        t = t.reshape(t.shape[0] * t.shape[1], t.shape[2], t.shape[3])
        t = self.temporal_conv1(t)
        t = t.reshape(x.shape[0], -1, t.shape[1], t.shape[2])
        
        # Graph convolution
        t = t.permute(0, 3, 1, 2)  # [batch_size, time_steps, out_channels, num_nodes]
        t = t.reshape(t.shape[0] * t.shape[1], t.shape[2], t.shape[3])
        t = self.graph_conv(t, A)
        t = t.reshape(x.shape[0], -1, t.shape[1], t.shape[2])
        
        # Temporal convolution 2
        t = t.permute(0, 2, 3, 1)  # [batch_size, out_channels, num_nodes, time_steps]
        t = t.reshape(t.shape[0] * t.shape[1], t.shape[2], t.shape[3])
        t = self.temporal_conv2(t)
        t = t.reshape(x.shape[0], -1, t.shape[1], t.shape[2])
        
        # Skip connection
        out = t + residual if residual.shape == t.shape else t
        out = self.relu(self.batch_norm(out))
        return out
    
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.W1 = nn.Linear(in_channels, in_channels, bias=False)
        self.W2 = nn.Linear(in_channels, in_channels, bias=False)
        self.W3 = nn.Linear(in_channels, 1, bias=False)
        
    def forward(self, x):
        # x shape: [batch_size, num_nodes, features]
        batch_size, num_nodes = x.shape[0], x.shape[1]
        
        # Compute spatial attention scores
        x1 = self.W1(x)  # [batch_size, num_nodes, features]
        x2 = self.W2(x)  # [batch_size, num_nodes, features]
        
        # Broadcast add for pairwise combination
        x1_expanded = x1.unsqueeze(2)  # [batch_size, num_nodes, 1, features]
        x2_expanded = x2.unsqueeze(1)  # [batch_size, 1, num_nodes, features]
        x_combined = torch.tanh(x1_expanded + x2_expanded)  # [batch_size, num_nodes, num_nodes, features]
        
        # Calculate attention matrix
        att_matrix = self.W3(x_combined).squeeze(-1)  # [batch_size, num_nodes, num_nodes]
        att_matrix = F.softmax(att_matrix, dim=-1)
        
        return att_matrix
    
class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers):
        super(GNNEncoder, self).__init__()
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
            
        self.gnn_layers.append(GCNConv(hidden_dim, latent_dim))
        self.relu = nn.ReLU()
        
    def forward(self, x, edge_index):
        for i, layer in enumerate(self.gnn_layers[:-1]):
            x = layer(x, edge_index)
            x = self.relu(x)
        
        x = self.gnn_layers[-1](x, edge_index)
        return x
        
class GNNDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, num_layers):
        super(GNNDecoder, self).__init__()
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(GCNConv(latent_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
            
        self.gnn_layers.append(GCNConv(hidden_dim, output_dim))
        self.relu = nn.ReLU()
        
    def forward(self, z, edge_index):
        x = z
        for i, layer in enumerate(self.gnn_layers[:-1]):
            x = layer(x, edge_index)
            x = self.relu(x)
        
        x = self.gnn_layers[-1](x, edge_index)
        return x

# 종합 모델 구현
class STGAT_AD(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_nodes, seq_len, pred_len):
        super(STGAT_AD, self).__init__()
        self.encoder = STGATEncoder(input_dim, hidden_dim, num_nodes, seq_len)
        self.decoder = STGATDecoder(hidden_dim, input_dim, num_nodes, pred_len)
        self.predictor = STGATPredictor(hidden_dim, input_dim, num_nodes, pred_len)
        
    def forward(self, x, adj_mat):
        # Encoding
        latent = self.encoder(x, adj_mat)
        
        # Reconstruction path
        recon = self.decoder(latent, adj_mat)
        
        # Prediction path
        pred = self.predictor(latent, adj_mat)
        
        return recon, pred, latent
        
    def anomaly_score(self, x, x_next, adj_mat):
        recon, pred, latent = self.forward(x, adj_mat)
        
        # Reconstruction error
        recon_error = F.mse_loss(recon, x, reduction='none')
        recon_error = recon_error.mean(dim=-1)  # Average over features
        
        # Prediction error
        pred_error = F.mse_loss(pred, x_next, reduction='none')
        pred_error = pred_error.mean(dim=-1)  # Average over features
        
        # Combine errors
        anomaly_scores = 0.7 * pred_error + 0.3 * recon_error
        
        return anomaly_scores, recon, pred
    
def train(model, train_loader, val_loader, optimizer, device, epochs=100):
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_x, batch_x_next, adj_mat in train_loader:
            batch_x = batch_x.to(device)
            batch_x_next = batch_x_next.to(device)
            adj_mat = adj_mat.to(device)
            
            optimizer.zero_grad()
            recon, pred, _ = model(batch_x, adj_mat)
            
            # Compute loss
            recon_loss = F.mse_loss(recon, batch_x)
            pred_loss = F.mse_loss(pred, batch_x_next)
            loss = 0.3 * recon_loss + 0.7 * pred_loss
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        val_loss = validate(model, val_loader, device)
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {val_loss:.6f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

def compute_anomaly_scores(model, test_loader, device):
    model.eval()
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_x_next, adj_mat, labels in test_loader:
            batch_x = batch_x.to(device)
            batch_x_next = batch_x_next.to(device)
            adj_mat = adj_mat.to(device)
            
            # Compute anomaly scores
            scores, _, _ = model.anomaly_score(batch_x, batch_x_next, adj_mat)
            
            all_scores.append(scores.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    
    return all_scores, all_labels
#%%
''''''