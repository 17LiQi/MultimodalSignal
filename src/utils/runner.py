# runner.py
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# --- 导入所有需要实例化的类 ---
from dataset.wesad_early_dataset import WesadEarlyFusionDataset
from dataset.wesad_feature_dataset import WesadFeatureDataset
from dataset.wesad_feature_sequence_dataset import WesadFeatureSequenceDataset
from dataset.wesad_hybrid_dataset import WesadHybridDataset
from dataset.wesad_hrv_dataset import WesadHrvDataset
from dataset.wesad_sequence_dataset import WesadSequenceDataset
from models.cnn_transformer_gru import CnnTransformerGru

from models.early_cnn_1d import SimpleCNN1D
from models.feature_gru import FeatureGRU
from models.hybrid import HybridModel
from models.early_resnet_1d import ResNet1D
from models.early_cnn_gru import CnnGruModel
from models.feature_mlp import MLP
from models.feature_tabnet import TabNetModelWrapper
from models.feature_xgboost import XGBoostModel
from models.transformer_encoder import TransformerClassifier
from models.feature_cnn_gru import FeatureCnnGru


from trainer.trainer import Trainer
from trainer.ml_trainer import MLTrainer
from trainer.tabnet_trainer import TabNetTrainer
from utils.path_manager import get_path_manager

def create_datasets(config, data_folder, train_subjects, val_subjects, test_subject):
    """根据配置创建训练、验证和测试数据集。"""
    path = get_path_manager()
    early_path = path.DATA_EARLY
    feature_path = path.DATA_FEATURE
    if config.fusion.type == 'early':
        DatasetClass = WesadEarlyFusionDataset
        train_ds = DatasetClass(data_folder, train_subjects, config.dataset.channels_to_use,
                                config.dataset.all_channel_names)
        val_ds = DatasetClass(data_folder, val_subjects, config.dataset.channels_to_use,
                              config.dataset.all_channel_names)
        test_ds = DatasetClass(data_folder, [test_subject], config.dataset.channels_to_use,
                               config.dataset.all_channel_names)
    elif config.fusion.type == 'feature':
        DatasetClass = WesadFeatureDataset
        train_ds = DatasetClass(data_folder, train_subjects)
        val_ds = DatasetClass(data_folder, val_subjects)
        test_ds = DatasetClass(data_folder, [test_subject])
    elif config.fusion.type == 'feature_sequence':
        # 特征序列融合流程
        DatasetClass = WesadFeatureSequenceDataset
        seq_params = config.dataset.sequence_params
        train_ds = DatasetClass(data_folder, train_subjects, seq_params.sequence_length, seq_params.step)
        val_ds = DatasetClass(data_folder, val_subjects, seq_params.sequence_length, seq_params.step)
        test_ds = DatasetClass(data_folder, [test_subject], seq_params.sequence_length, seq_params.step)
    elif config.fusion.type == 'hybrid':
        DatasetClass = WesadHybridDataset

        seq_params = config.dataset.sequence_params

        train_ds = DatasetClass(early_path, feature_path, train_subjects, config.dataset.channels_to_use,
                                config.dataset.all_channel_names, seq_params.sequence_length, seq_params.step)
        val_ds = DatasetClass(early_path, feature_path, val_subjects, config.dataset.channels_to_use,
                              config.dataset.all_channel_names, seq_params.sequence_length, seq_params.step)
        test_ds = DatasetClass(early_path, feature_path, [test_subject], config.dataset.channels_to_use,
                               config.dataset.all_channel_names, seq_params.sequence_length, seq_params.step)
    elif config.fusion.type == 'hrv_sequence':
        DatasetClass = WesadHrvDataset
        data_folder = path.DATA_ROOT  / 'hrv_sequence'
        train_ds = DatasetClass(data_folder, train_subjects)
        val_ds = DatasetClass(data_folder, val_subjects)
        test_ds = DatasetClass(data_folder, [test_subject])

    elif config.fusion.type == 'sequence':
        DatasetClass = WesadSequenceDataset
        data_folder = path.DATA_ROOT / 'wesad_sequence_features'
        seq_params = config.dataset.sequence_params

        train_ds = DatasetClass(data_folder, train_subjects, seq_params.sequence_length, seq_params.step)
        val_ds = DatasetClass(data_folder, val_subjects, seq_params.sequence_length, seq_params.step)
        test_ds = DatasetClass(data_folder, [test_subject], seq_params.sequence_length, seq_params.step)
    else:
        raise ValueError(f"未知的融合类型: {config.fusion.type}")
    return train_ds, val_ds, test_ds


def create_model(config):
    """根据配置创建模型实例。"""
    model_name = config.model.name

    # 1. 提取所有模型共享的核心参数
    num_classes = config.model.num_classes

    # 2. 提取当前模型专属的结构超参数
    model_args = dict(config.model.get('params', {}))

    # 3. 根据融合类型和模型名称进行实例化

    # ******************* early ***************************** #
    if config.fusion.type == 'early':
        in_channels = len(config.dataset.channels_to_use)
        if model_name == "cnn_1d":
            return SimpleCNN1D(in_channels, num_classes, **model_args)
        if model_name == "resnet_1d":
            return ResNet1D(in_channels, num_classes, **model_args)
        if model_name == "cnn_gru":
            return CnnGruModel(in_channels, num_classes, **model_args)
        if model_name == "transformer":
            return TransformerClassifier(in_channels, num_classes, **model_args)

    # *********************** feature *********************** #
    elif config.fusion.type in ['feature', 'feature_sequence']:
        in_features = model_args.pop('in_features')  # 特征数是这类模型的核心参数

        if model_name == "mlp":
            # 确保 hidden_layers 是整数列表
            hidden_layers_cfg = model_args.get('hidden_layers', [])
            # 使用列表推导式将每个元素转换为整数
            model_args['hidden_layers'] = [int(h) for h in hidden_layers_cfg]
            return MLP(in_features=in_features, num_classes=num_classes, **model_args)
        if model_name == "feature_gru":
            return FeatureGRU(in_features=in_features, num_classes=num_classes, **model_args)
        if model_name == 'feature_cnn_gru':
            return FeatureCnnGru(in_features=in_features, num_classes=num_classes, **model_args)
        if model_name == "xgboost":
            return XGBoostModel(num_classes=num_classes, params=model_args)
        if model_name == "tabnet":
            return TabNetModelWrapper(
                input_dim=in_features, output_dim=num_classes,
                optimizer_params=dict(config.trainer.optimizer_params),
                scheduler_params=dict(config.trainer.scheduler_params),
                **model_args
            )

    # *********************** hybrid *********************** #
    elif config.fusion.type == 'hybrid':
        if model_name == 'hybrid':
            # --- 创建子模型 ---
            raw_conf = model_args.pop('raw_model')
            raw_in_channels = len(config.dataset.channels_to_use)
            # 递归或辅助调用来创建子模型
            raw_model = CnnGruModel(raw_in_channels, num_classes, **dict(raw_conf.params))

            feat_conf = model_args.pop('feature_model')
            feat_in_features = feat_conf.params.in_features
            feat_params = dict(feat_conf.params)
            feat_params.pop('in_features', None)
            feat_model = FeatureGRU(feat_in_features, num_classes, **dict(**feat_params))

            return HybridModel(raw_model, feat_model, num_classes, **model_args)

    # *********************** hrv_sequence *********************** #
    elif config.fusion.type == 'hrv_sequence':
        # HRV 序列是单通道的
        in_channels = 1
        num_classes = config.model.num_classes
        model_name = config.model.name
        model_args = dict(config.model.get('params', {}))

        if model_name == "cnn_gru":
            return CnnGruModel(in_channels, num_classes, **model_args)

        if model_name == "transformer":
            return TransformerClassifier(in_channels, num_classes, **model_args)

        if model_name == "cnn_transformer_gru":
            return CnnTransformerGru(in_channels, num_classes, **model_args)

    elif config.fusion.type == 'sequence':
        in_features = config.model.in_features
        model_args = dict(config.model.get('params', {}))
        if model_name == "transformer":
            return TransformerClassifier(in_features, num_classes, **model_args)

    raise ValueError(f"无法为配置创建模型: fusion='{config.fusion.type}', model='{model_name}'")


def run_one_fold(config, test_subject, all_subjects, paths, run_output_dir):
    """运行单次留一法交叉验证的折叠 (重构后的核心逻辑)。"""

    # 1. 设置路径和划分受试者
    processed_path = paths.DATA_ROOT
    fold_output_dir = run_output_dir / f'fold_test_on_{test_subject}'
    fold_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n===== 开始处理折叠: 测试集=[{test_subject}] =====")
    train_val_subjects = [s for s in all_subjects if s != test_subject]
    train_subjects, val_subjects = train_test_split(
        train_val_subjects, test_size=config.dataset.val_split_ratio, random_state=config.seed
    )

    # 2. 创建数据集
    data_folder = processed_path / config.dataset.name
    train_ds, val_ds, test_ds = create_datasets(config, data_folder, train_subjects, val_subjects, test_subject)

    # 3. 创建模型
    model = create_model(config)

    # 4. 根据模型类型，选择并执行对应的训练/评估流程
    model_name = config.model.name

    if model_name in ["cnn_1d", "resnet_1d", "cnn_gru", "mlp", "feature_gru", "hybrid", "transformer","feature_cnn_gru", "hrv_sequence", "cnn_transformer_gru"]:
        # PyTorch 流程
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        train_loader = DataLoader(train_ds, batch_size=config.trainer.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.trainer.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=config.trainer.batch_size, shuffle=False)

        # 实例化 PyTorch Trainer
        trainer = Trainer(model, config, fold_output_dir, train_ds)

        # 开始训练和评估
        trainer.train(train_loader, val_loader)
        _, test_acc, test_f1 = trainer.evaluate(test_loader, is_test=True)

    elif model_name == "xgboost":
        # XGBoost 流程
        trainer = MLTrainer(model, config, fold_output_dir)
        test_acc, test_f1 = trainer.train_and_evaluate(train_ds, val_ds, test_ds, test_subject)

    elif model_name == "tabnet":
        # TabNet 流程
        trainer = TabNetTrainer(model, config, fold_output_dir)
        test_acc, test_f1 = trainer.train_and_evaluate(train_ds, val_ds, test_ds)

    else:
        raise ValueError(f"未知的模型名称: {model_name}")

    return test_acc, test_f1
