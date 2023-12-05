# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 12:30:52 2022

@author: siahn(2022254004)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def k_means_algorithm(x1, cluster_count, max_index):
    plt.style.use('default')
    k_means = KMeans(init="k-means++", n_clusters=cluster_count, n_init=max_index, random_state=0)
    k_means.fit(x1)
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_
    print("==================== K-Means Result ====================")
    for cnt in range(len(x1)):
        print('좌표 :', x1[cnt], ', 그룹 :', k_means_labels[cnt])
    print("--------------------------------------------------------")
    for cnt in range(len(k_means_cluster_centers)):
        print('그룹', str(cnt + 1), '중심점 :', k_means_cluster_centers[cnt])
    print("========================================================")

    # 지정된 크기로 초기화
    fig = plt.figure(figsize=(12, 9))

    # 레이블 수에 따라 색상 배열 생성, 고유한 색상을 얻기 위해 set(k_means_labels) 설정
    colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

    # plot 생성
    ax = fig.add_subplot(1, 1, 1)
    color = []

    # 중심 정의
    for k, col in zip(range(cluster_count), colors):
        # 중심 마커 색 순서대로 정의
        color.append(col)

        cluster_center = k_means_cluster_centers[k]

        group_text = "Group" + str(k + 1)
        center_text = group_text + " center : " + str(cluster_center)

        # 중심 그리기
        ax.plot(k_means_cluster_centers[k][0], k_means_cluster_centers[k][1], 'o',
                markerfacecolor=col, markeredgecolor='k', markersize=10, label=center_text)
        ax.text(cluster_center[0], cluster_center[1], group_text)

    # 마커 그리기
    for i in range(len(x1)):
        clr = color[k_means_labels[i]]
        ax.plot(x1[i][0], x1[i][1], 'w', markerfacecolor=clr, marker='o', markeredgecolor='k',
                markersize=5)

    # 그래프 설정
    ax.set_title('Seong-in Ahn(2022254004) : K-Means Result')
    ax.set_xticks(np.arange(0, 10.5, 0.5),
                  labels=[0, '', 1, '', 2, '', 3, '', 4, '', 5, '', 6, '', 7, '', 8, '', 9, '', 10])
    ax.set_yticks(np.arange(0, 10.5, 0.5),
                  labels=[0, '', 1, '', 2, '', 3, '', 4, '', 5, '', 6, '', 7, '', 8, '', 9, '', 10])

    ax.legend(loc='upper left')

    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    X1 = np.array(
        [[0, 0], [1, 0], [0, 1], [1, 1], [2, 1], [1, 2], [2, 2], [3, 2], [6, 6], [7, 6],
         [8, 6], [6, 7], [7, 7], [8, 7], [9, 7], [7, 8], [8, 8], [9, 8], [8, 9], [9, 9]])
    print("========================================================")
    print("과제명 : K-means-algorithm")
    print("학번 : 2022254004")
    print("이름 : 안성인")
    print("--------------------------------------------------------")
    K = int(input("형성할 클러스터 수 : "))
    max_iter = int(input("반복 실행 횟수 : "))
    k_means_algorithm(X1, K, max_iter)
