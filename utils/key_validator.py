import os
import queue
import random
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

from common.Logger import logger
from common.config import Config
from common.translations import get_translator
from utils.file_manager import file_manager
from utils.sync_utils import sync_utils

# 获取翻译函数
t = get_translator().t


class PendingKey:
    """待验证密钥信息"""
    def __init__(self, key: str, repo_name: str, file_path: str, file_url: str):
        self.key = key
        self.repo_name = repo_name
        self.file_path = file_path
        self.file_url = file_url
        self.timestamp = time.time()


class KeyValidator:
    """异步密钥验证管理器"""

    def __init__(self, max_workers: int = 5):
        """
        初始化密钥验证管理器
        
        Args:
            max_workers: 最大并发验证线程数
        """
        self.validation_queue = queue.Queue()
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="KeyValidator")
        self.shutdown_flag = False
        
        # 验证结果统计
        self.stats = {
            "total_queued": 0,
            "total_validated": 0,
            "valid_keys": 0,
            "rate_limited_keys": 0,
            "invalid_keys": 0,
            "paid_keys": 0
        }
        self.stats_lock = threading.Lock()
        
        # 按文件分组的验证结果（用于批量保存）
        self.results_by_file: Dict[str, Dict[str, List[str]]] = {}
        self.results_lock = threading.Lock()
        
        # 启动验证工作线程
        for i in range(max_workers):
            self.executor.submit(self._validation_worker, i)
        
        logger.info(f"🚀 异步密钥验证器已启动，并发数: {max_workers}")

    def add_key(self, key: str, repo_name: str, file_path: str, file_url: str) -> None:
        """
        添加密钥到验证队列
        
        Args:
            key: API密钥
            repo_name: 仓库名称
            file_path: 文件路径
            file_url: 文件URL
        """
        pending_key = PendingKey(key, repo_name, file_path, file_url)
        self.validation_queue.put(pending_key)
        
        with self.stats_lock:
            self.stats["total_queued"] += 1

    def _build_headers(self, api_key: str) -> Dict[str, str]:
        """构造 API 请求头，别乱加奇怪字段。"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "HajimiKing/1.0",
        }
        return headers

    def _perform_request(
        self,
        url: str,
        api_key: str,
        method: str = "GET",
        json_payload: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[requests.Response], Optional[str]]:
        """执行带鉴权请求，返回 (响应, 错误码)。"""
        proxies = Config.get_random_proxy()
        headers = self._build_headers(api_key)

        try:
            if json_payload is not None:
                headers.setdefault("Content-Type", "application/json")

            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=json_payload,
                timeout=Config.CUSTOM_API_TIMEOUT,
                proxies=proxies,
            )
            return response, None
        except requests.exceptions.Timeout:
            return None, "timeout"
        except requests.exceptions.ProxyError:
            return None, "proxy_error"
        except requests.exceptions.ConnectionError:
            return None, "connection_error"
        except requests.exceptions.RequestException as exc:
            return None, f"error:{exc.__class__.__name__}"

    def _validate_api_key(self, api_key: str) -> Union[str, str]:
        """验证是否能打通自定义 API 基础入口。"""
        base_url = Config.CUSTOM_API_BASE
        if not base_url:
            return "invalid_api_base"

        try:
            time.sleep(random.uniform(0.5, 1.5))
            check_model = Config.CUSTOM_CHECK_MODEL or Config.CUSTOM_PAID_MODEL
            if not check_model:
                logger.warning("⚠️ 未配置 CUSTOM_CHECK_MODEL 或 CUSTOM_PAID_MODEL，无法通过 chat/completions 验证密钥")
                return "missing_check_model"

            payload = {
                "model": check_model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 8,
                "stream": False,
            }
            response, error = self._perform_request(f"{base_url}/chat/completions", api_key, method="POST", json_payload=payload)
            if error:
                return error

            status = response.status_code
            if status in (200, 201):
                try:
                    data = response.json()
                except ValueError:
                    logger.error(f"🔥 API密钥验证返回非 JSON: status={status}")
                    return "error:invalid_json"

                if not isinstance(data, dict):
                    logger.error(f"🔥 API密钥验证返回异常结构: status={status}, type={type(data)}")
                    return "error:invalid_payload"

                if data.get("error"):
                    error_info = data["error"]
                    if isinstance(error_info, dict):
                        code = error_info.get("code") or error_info.get("type") or "unknown_error"
                        return f"error:{code}"
                    return f"error:{error_info}"

                if not data.get("choices"):
                    logger.error(f"🔥 API密钥验证缺少 choices 字段: status={status}")
                    return "error:missing_choices"
                return "ok"
            if status == 401:
                return "not_authorized_key"
            if status == 403:
                return "disabled"
            if status == 404:
                return "model_not_found"
            if status == 429:
                return "rate_limited"
            return f"http_{status}"
        except Exception as exc:  # noqa: BLE001
            logger.error(f"🔥 API密钥验证崩了: {exc}")
            return "error:unexpected"

    def _validate_paid_model_key(self, api_key: str) -> Union[str, str]:
        """验证密钥是否支持付费模型。"""
        paid_model = Config.CUSTOM_PAID_MODEL
        if not paid_model:
            return "skip"

        base_url = Config.CUSTOM_API_BASE
        if not base_url:
            return "invalid_api_base"

        try:
            time.sleep(random.uniform(0.5, 1.5))
            payload = {
                "model": paid_model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 8,
                "stream": False,
            }
            response, error = self._perform_request(f"{base_url}/chat/completions", api_key, method="POST", json_payload=payload)
            if error:
                return error

            status = response.status_code
            if status in (200, 201):
                try:
                    data = response.json()
                except ValueError:
                    logger.error(f"🔥 付费模型验证返回非 JSON: status={status}")
                    return "error:invalid_json"

                if not isinstance(data, dict):
                    logger.error(f"🔥 付费模型验证返回异常结构: status={status}, type={type(data)}")
                    return "error:invalid_payload"

                if data.get("error"):
                    error_info = data["error"]
                    if isinstance(error_info, dict):
                        code = error_info.get("code") or error_info.get("type") or "unknown_error"
                        return f"error:{code}"
                    return f"error:{error_info}"

                if not data.get("choices"):
                    logger.error(f"🔥 付费模型验证缺少 choices 字段: status={status}")
                    return "error:missing_choices"
                return "ok"
            if status == 401:
                return "not_authorized_for_paid"
            if status == 403:
                return "disabled"
            if status == 404:
                return "model_not_found"
            if status == 429:
                return "rate_limited"
            return f"http_{status}"
        except Exception as exc:  # noqa: BLE001
            logger.error(f"🔥 付费模型验证崩了: {exc}")
            return "error:unexpected"

    def _validation_worker(self, worker_id: int) -> None:
        """
        验证工作线程
        
        Args:
            worker_id: 工作线程ID
        """
        logger.info(f"🔧 验证工作线程 #{worker_id} 已启动")
        
        while not self.shutdown_flag:
            try:
                # 从队列获取待验证密钥，超时5秒
                try:
                    pending_key = self.validation_queue.get(timeout=5)
                except queue.Empty:
                    continue
                
                key = pending_key.key
                repo_name = pending_key.repo_name
                file_path = pending_key.file_path
                file_url = pending_key.file_url
                
                # 执行验证
                validation_result = self._validate_api_key(key)
                
                # 初始化结果存储
                file_key = f"{repo_name}::{file_path}"
                with self.results_lock:
                    if file_key not in self.results_by_file:
                        self.results_by_file[file_key] = {
                            "repo_name": repo_name,
                            "file_path": file_path,
                            "file_url": file_url,
                            "valid_keys": [],
                            "rate_limited_keys": [],
                            "paid_keys": []
                        }
                
                # 处理验证结果
                if validation_result == "ok":
                    # 有效密钥
                    logger.info(t('valid_key', key))
                    
                    with self.results_lock:
                        self.results_by_file[file_key]["valid_keys"].append(key)
                    
                    with self.stats_lock:
                        self.stats["valid_keys"] += 1
                    
                    # 对有效密钥进行付费模型验证
                    logger.info(f"🔍 正在验证付费模型: {key[:20]}...")
                    paid_validation_result = self._validate_paid_model_key(key)
                    if paid_validation_result == "ok":
                        logger.info(f"💎 付费密钥验证成功: {key[:20]}... (支持{Config.CUSTOM_PAID_MODEL})")
                        
                        with self.results_lock:
                            self.results_by_file[file_key]["paid_keys"].append(key)
                        
                        with self.stats_lock:
                            self.stats["paid_keys"] += 1
                    elif paid_validation_result == "skip":
                        logger.info(f"🧪 付费模型验证跳过: {key[:20]}... (未配置 CUSTOM_PAID_MODEL)")
                    else:
                        logger.info(f"ℹ️ 付费模型验证失败: {key[:20]}... ({paid_validation_result})")
                
                elif "rate_limited" in validation_result:
                    # 限速密钥
                    logger.warning(t('rate_limited_key', key, validation_result))
                    
                    # 根据RATE_LIMITED_HANDLING配置决定如何处理429密钥
                    handling = Config.RATE_LIMITED_HANDLING.strip().lower()
                    
                    if handling == "discard":
                        logger.info(f"⏰❌ 429密钥已丢弃: {key[:20]}... (RATE_LIMITED_HANDLING=discard)")
                    elif handling == "save_only":
                        with self.results_lock:
                            self.results_by_file[file_key]["rate_limited_keys"].append(key)
                        logger.info(f"⏰💾 429密钥仅本地保存: {key[:20]}... (RATE_LIMITED_HANDLING=save_only)")
                    elif handling == "sync":
                        with self.results_lock:
                            self.results_by_file[file_key]["rate_limited_keys"].append(key)
                            self.results_by_file[file_key]["valid_keys"].append(key)
                        logger.info(f"⏰✅ 429密钥视为正常密钥: {key[:20]}... (RATE_LIMITED_HANDLING=sync)")
                    elif handling == "sync_separate":
                        with self.results_lock:
                            self.results_by_file[file_key]["rate_limited_keys"].append(key)
                        logger.info(f"⏰🔄 429密钥将同步到独立分组: {key[:20]}... (RATE_LIMITED_HANDLING=sync_separate)")
                    else:
                        with self.results_lock:
                            self.results_by_file[file_key]["rate_limited_keys"].append(key)
                        logger.warning(f"⏰ 未知的RATE_LIMITED_HANDLING值: {handling}，使用默认行为(save_only)")
                    
                    with self.stats_lock:
                        self.stats["rate_limited_keys"] += 1
                else:
                    # 无效密钥
                    logger.info(t('invalid_key', key, validation_result))
                    
                    with self.stats_lock:
                        self.stats["invalid_keys"] += 1
                
                # 更新已验证计数
                with self.stats_lock:
                    self.stats["total_validated"] += 1
                
                # 标记任务完成
                self.validation_queue.task_done()
                
            except Exception as e:
                logger.error(f"❌ 验证工作线程 #{worker_id} 发生错误: {e}")
                traceback.print_exc()
                try:
                    self.validation_queue.task_done()
                except:
                    pass
        
        logger.info(f"🔧 验证工作线程 #{worker_id} 已停止")

    def flush_results(self) -> Tuple[int, int, int]:
        """
        刷新所有验证结果到文件和同步队列
        
        Returns:
            tuple: (valid_keys_count, rate_limited_keys_count, paid_keys_count)
        """
        total_valid = 0
        total_rate_limited = 0
        total_paid = 0
        
        with self.results_lock:
            for file_key, results in self.results_by_file.items():
                repo_name = results["repo_name"]
                file_path = results["file_path"]
                file_url = results["file_url"]
                valid_keys = results["valid_keys"]
                rate_limited_keys = results["rate_limited_keys"]
                paid_keys = results["paid_keys"]
                
                # 保存有效密钥
                if valid_keys:
                    file_manager.save_valid_keys(repo_name, file_path, file_url, valid_keys)
                    logger.info(t('saved_valid_keys', len(valid_keys)))
                    
                    # 添加到同步队列
                    try:
                        sync_utils.add_keys_to_queue(valid_keys)
                        logger.info(t('added_to_queue', len(valid_keys)))
                    except Exception as e:
                        logger.error(t('error_adding_to_queue', e))
                    
                    total_valid += len(valid_keys)
                
                # 保存限速密钥
                if rate_limited_keys:
                    file_manager.save_rate_limited_keys(repo_name, file_path, file_url, rate_limited_keys)
                    logger.info(t('saved_rate_limited_keys', len(rate_limited_keys)))
                    
                    # 根据配置决定是否将429密钥同步到独立分组
                    if Config.RATE_LIMITED_HANDLING.strip().lower() == "sync_separate":
                        try:
                            sync_utils.add_rate_limited_keys_to_queue(rate_limited_keys)
                            logger.info(f"⏰ 已添加 {len(rate_limited_keys)} 个429密钥到独立上传队列")
                        except Exception as e:
                            logger.error(f"⏰ 添加429密钥到队列时出错: {e}")
                    
                    total_rate_limited += len(rate_limited_keys)
                
                # 保存付费密钥
                if paid_keys:
                    file_manager.save_paid_keys(repo_name, file_path, file_url, paid_keys)
                    logger.info(f"💎 已保存付费密钥: {len(paid_keys)} 个")
                    
                    # 根据配置决定是否上传付费密钥到GPT-load
                    if Config.parse_bool(Config.GPT_LOAD_PAID_SYNC_ENABLED):
                        try:
                            sync_utils.add_paid_keys_to_queue(paid_keys)
                            logger.info(f"💎 已添加 {len(paid_keys)} 个付费密钥到上传队列")
                        except Exception as e:
                            logger.error(f"💎 添加付费密钥到队列时出错: {e}")
                    else:
                        logger.info(f"💎 付费密钥上传功能已关闭，仅本地保存 {len(paid_keys)} 个密钥")
                    
                    total_paid += len(paid_keys)
            
            # 清空结果缓存
            self.results_by_file.clear()
        
        return total_valid, total_rate_limited, total_paid

    def get_queue_size(self) -> int:
        """获取待验证队列大小"""
        return self.validation_queue.qsize()

    def get_stats(self) -> Dict[str, int]:
        """获取验证统计信息"""
        with self.stats_lock:
            return self.stats.copy()

    def wait_completion(self, timeout: Optional[float] = None) -> bool:
        """
        等待所有待验证密钥完成
        
        Args:
            timeout: 超时时间（秒），None表示无限等待
            
        Returns:
            bool: True表示所有任务完成，False表示超时
        """
        queue_size = self.get_queue_size()
        if queue_size > 0:
            logger.info(f"⏳ 等待 {queue_size} 个密钥验证完成...")
        
        start_time = time.time()
        last_log_time = start_time
        
        while True:
            queue_size = self.get_queue_size()
            
            # 队列为空，任务完成
            if queue_size == 0:
                # 再等待一小段时间确保所有工作线程处理完成
                time.sleep(1)
                if self.get_queue_size() == 0:
                    logger.info("✅ 所有密钥验证完成")
                    return True
            
            # 检查超时
            if timeout is not None and (time.time() - start_time) > timeout:
                logger.warning(f"⚠️ 等待验证完成超时，剩余 {queue_size} 个密钥")
                return False
            
            # 每10秒输出一次进度
            if time.time() - last_log_time >= 10:
                stats = self.get_stats()
                logger.info(f"⏳ 验证进度: {stats['total_validated']}/{stats['total_queued']} 已完成，剩余 {queue_size} 个待验证")
                last_log_time = time.time()
            
            time.sleep(1)

    def reset_stats(self) -> None:
        """重置统计信息"""
        with self.stats_lock:
            self.stats = {
                "total_queued": 0,
                "total_validated": 0,
                "valid_keys": 0,
                "rate_limited_keys": 0,
                "invalid_keys": 0,
                "paid_keys": 0
            }

    def shutdown(self) -> None:
        """关闭验证器"""
        logger.info("🛑 正在关闭异步密钥验证器...")
        
        # 等待所有任务完成
        self.wait_completion(timeout=60)
        
        # 设置关闭标志
        self.shutdown_flag = True
        
        # 关闭线程池
        self.executor.shutdown(wait=True)
        
        # 刷新剩余结果
        self.flush_results()
        
        logger.info("✅ 异步密钥验证器已关闭")


# 创建全局实例（从配置读取并发数）
key_validator = KeyValidator(max_workers=Config.KEY_VALIDATOR_MAX_WORKERS)



