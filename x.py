import torch
import os


def merge_and_save_tensors(total_count=42030, batch_size=4000, save_path="data/tensor/feature_maps",
                           base_path=r"C:\Users\wlstn\.cache\kagglehub\datasets\tensor\feature_maps"):
    os.makedirs(save_path, exist_ok=True)  # 저장 폴더 생성

    tensors = []
    merged_index = 0

    for i in range(total_count + 1):
        tensor_path = os.path.join(base_path, f"feature_map_{i}.pt")
        tensor = torch.load(tensor_path)  # 텐서 로드
        tensors.append(tensor)

        if len(tensors) == batch_size:
            merged_tensor = torch.cat(tensors, dim=0)  # (200000, 765) 생성
            torch.save(merged_tensor, os.path.join(save_path, f"feature_map_{merged_index}.pt"))
            tensors = []  # 리스트 초기화
            merged_index += 1

    # 남은 텐서가 있을 경우 저장
    if tensors:
        merged_tensor = torch.cat(tensors, dim=0)
        torch.save(merged_tensor, os.path.join(save_path, f"merged_{merged_index}.pt"))


merge_and_save_tensors()