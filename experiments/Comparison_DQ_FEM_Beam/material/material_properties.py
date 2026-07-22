"""
石墨烯增强功能梯度材料(GPL-FGM)属性计算模块

该模块实现了石墨烯平板片(GPL)增强铜基复合材料的属性计算。

核心功能：
1. 材料常数定义：铜(Cu)和石墨烯(Graphene)的基本属性
2. 修正因子计算：基于半经验公式考虑温度和形状因子的影响
3. 层向属性计算：使用修正的混合规则计算各层的等效属性
4. 刚度系数计算：通过层合板理论计算A11、B11、D11、A55
5. 无量纲化处理：以纯铜属性为参考进行无量纲化
6. 热力学参数：计算热载荷引起的轴力N_xT和弯矩M_xT

材料分布类型：
- X型：GPL在两端多、中间少，增强抗弯能力
- U型：GPL均匀分布，各层属性一致
- O型：GPL在中间多、两端少，增强抗剪能力

基于Timoshenko梁理论(FSDT)，适配ODIL求解器框架
"""

from typing import Tuple, Dict, NamedTuple, Optional
from dataclasses import dataclass
import numpy as np
import warnings

# 兼容不同版本的SciPy - 这个模块主要使用数值计算，不需要复杂积分

@dataclass
class MaterialConstants:
    """材料基本常数类

    定义基体材料(铜)和增强相(石墨烯)的基本属性。
    所有值基于实验数据和文献报道。
    """
    # 参考温度（室温，用于计算温度变化）
    T_0: float = 300.0  # K

    # 铜(Cu)的材料属性 - 基体材料
    E_Cu: float = 65.79e9    # 杨氏模量 (Pa) - 描述材料的拉伸刚度
    nu_Cu: float = 0.387     # 泊松比 - 横向应变与纵向应变之比
    rho_Cu: float = 8.80e3   # 密度 (kg/m³) - 影响惯性力
    alpha_Cu: float = 16.51e-6  # 热膨胀系数 (1/K) - 温度变化引起的应变

    # 石墨烯(Graphene)的材料属性 - 增强相
    E_Gr: float = 929.57e9   # 杨氏模量 (Pa) - 远高于铜，提供增强效果
    nu_Gr: float = 0.220     # 泊松比 - 低于铜
    rho_Gr: float = 1.80e3   # 密度 (kg/m³) - 远低于铜，有利于轻量化
    alpha_Gr: float = -3.98e-6  # 热膨胀系数 (1/K) - 负值，热缩冷胀

    # GPL(石墨烯平板片)几何参数 - 用于Halpin-Tsai模型
    l_Gr: float = 83.76  # GPL长度(无量纲) - 影响长径比
    t_Gr: float = 3.4    # GPL厚度(无量纲) - 影响增强效率

class MaterialProperties(NamedTuple):
    """材料属性结果类"""
    E: float        # 有效杨氏模量
    nu: float       # 有效泊松比
    alpha: float    # 有效热膨胀系数
    rho: float      # 有效密度
    layer_properties: Dict  # 各层详细属性
    Q11: Dict       # 各层Q11刚度系数
    Q55: Dict       # 各层Q55刚度系数

class DimensionlessParameters(NamedTuple):
    """无量纲参数结果类"""
    a11: float      # 无量纲拉伸刚度
    b11: float      # 无量纲弯拉耦合刚度
    d11: float      # 无量纲弯曲刚度
    a55: float      # 无量纲剪切刚度
    lambda_val: float  # 长高比
    n_xT: float     # 无量纲轴向热力
    m_xT: float     # 无量纲弯曲热力矩

class MaterialCalculator:
    """材料属性计算器"""
    
    def __init__(self, constants: Optional[MaterialConstants] = None):
        """
        初始化材料计算器
        
        参数:
            constants: 材料常数，如果为None则使用默认值
        """
        self.constants = constants if constants is not None else MaterialConstants()
    
    def compute_correction_factors(self, V_Gr: float, T_ratio: float, H_Gr: float) -> Tuple[float, float, float, float]:
        """
        计算材料属性的修正因子

        基于半经验公式，考虑温度和GPL形状对复合材料属性的影响。
        这些公式来源于实验数据拟合和分子动力学模拟。

        参数:
            V_Gr: GPL体积分数 (0-1)
            T_ratio: 温度比T/T₀ (用于考虑温度效应)
            H_Gr: GPL形状因子 (描述GPL的长径比和取向)

        返回:
            f_E: 杨氏模量修正因子
            f_nu: 泊松比修正因子
            f_alpha: 热膨胀系数修正因子
            f_rho: 密度修正因子
        """
        if V_Gr == 0:
            # 纯铜情况
            return 1.0, 1.0, 1.0, 1.0
        
        # 确保参数在合理范围内
        V_Gr = np.clip(V_Gr, 0.0, 1.0)
        H_Gr = np.clip(H_Gr, 0.0, 10.0)
        T_ratio = np.clip(T_ratio, 0.5, 3.0)
        
        # 杨氏模量修正因子 - 考虑V_Gr、T和H_Gr的耦合效应
        f_E = (1.11 - 1.22 * V_Gr - 0.134 * T_ratio + 0.559 * V_Gr * T_ratio
               - 5.5 * H_Gr * V_Gr + 38 * H_Gr * V_Gr ** 2 - 20.6 * H_Gr ** 2 * V_Gr ** 2)

        # 泊松比修正因子 - 可能为负值，这是物理上合理的
        f_nu = (1.01 - 1.43 * V_Gr + 0.165 * T_ratio - 16.8 * H_Gr * V_Gr
                - 1.1 * H_Gr * V_Gr * T_ratio + 16 * H_Gr ** 2 * V_Gr ** 2)

        # 热膨胀系数修正因子 - 反映GPL的负热膨胀特性
        f_alpha = 0.794 - 16.8 * V_Gr ** 2 - 0.0279 * T_ratio * (1 + V_Gr)

        # 密度修正因子 - 通常接近1，因为GPL密度远低于铜
        f_rho = 1.01 - 2.01 * V_Gr ** 2 - 0.0131 * T_ratio
        
        # # 确保修正因子为正值（除了f_nu，允许负值以匹配MATLAB）
        # f_E = max(f_E, 0.00001)
        # # f_nu 可以为负值！这是关键差异
        # f_alpha = max(f_alpha, 0.00001)
        # f_rho = max(f_rho, 0.00001)
        
        return f_E, f_nu, f_alpha, f_rho
    
    def compute_layer_properties(self, layer_index: int, total_layers: int,
                               V_Gr_base: float, T_ratio: float, H_Gr: float,
                               distribution_type: str = 'X') -> Tuple[float, float, float, float]:
        """
        计算第layer_index层的材料属性

        根据分布类型调整各层的GPL体积分数，实现功能梯度分布。
        使用修正的混合规则和几何参数计算等效材料属性。

        参数:
            layer_index: 层编号 (1开始，从底到顶)
            total_layers: 总层数 (通常为10)
            V_Gr_base: 平均GPL体积分数
            T_ratio: 温度比T/T₀
            H_Gr: GPL形状因子
            distribution_type: 分布类型
                'X': 两端富GPL (抗弯优化)
                'O': 中间富GPL (抗剪优化)
                'U': 均匀分布

        返回:
            该层的等效性质: E, nu, alpha, rho
        """
        # 计算该层的石墨烯体积分数
        if total_layers == 1:
            V_Gr_layer = V_Gr_base
        else:
            k = layer_index  # 使用1-based索引
            N = total_layers
            
            if distribution_type.upper() == 'X':
                # X型分布：两端富GPL，中性面附近贫GPL
                # 有利于提高弯曲刚度D11
                V_Gr_layer = 2 * V_Gr_base * abs(2 * k - N - 1) / N
            elif distribution_type.upper() == 'O':
                # O型分布：中性面附近富GPL，两端贫GPL
                # 有利于提高剪切刚度A55
                V_Gr_layer = 2 * V_Gr_base * (1 - abs(2 * k - N - 1) / N)
            elif distribution_type.upper() == 'U':
                # U型分布：各层GPL含量相同
                # 作为参考基准
                V_Gr_layer = V_Gr_base
            else:
                raise ValueError(f"不支持的分布类型: {distribution_type}")
        
        # 确保体积分数在有效范围内
        V_Gr_layer = np.clip(V_Gr_layer, 0.0, 1.0)
        V_Cu_layer = 1.0 - V_Gr_layer
        
        # 计算修正因子
        f_E, f_nu, f_alpha, f_rho = self.compute_correction_factors(V_Gr_layer, T_ratio, H_Gr)
        
        # 计算几何参数
        ksai = 2 * (self.constants.l_Gr / self.constants.t_Gr)
        eta = ((self.constants.E_Gr / self.constants.E_Cu - 1) / 
               (self.constants.E_Gr / self.constants.E_Cu + ksai))
        
        # 计算材料属性
        # 杨氏模量
        E = ((1 + ksai * eta * V_Gr_layer) / (1 - eta * V_Gr_layer) * 
             self.constants.E_Cu * f_E)
        
        # 泊松比
        nu = ((self.constants.nu_Gr * V_Gr_layer + 
               self.constants.nu_Cu * V_Cu_layer) * f_nu)
        
        # 热膨胀系数
        alpha = ((self.constants.alpha_Gr * V_Gr_layer + 
                  self.constants.alpha_Cu * V_Cu_layer) * f_alpha)
        
        # 密度
        rho = ((self.constants.rho_Gr * V_Gr_layer + 
                self.constants.rho_Cu * V_Cu_layer) * f_rho)
        
        # 检查结果的合理性
        if E <= 0 or nu >= 1 or rho <= 0:
            warnings.warn(f"第{layer_index}层材料属性异常: E={E:.2e}, nu={nu:.3f}, rho={rho:.2e}")
        
        return E, nu, alpha, rho
    
    def compute_material_properties(self, h: float, num_layers: int, 
                                  W_Gr: float = 0.025, H_Gr: float = 0.8, 
                                  T: float = 300.0, distribution_type: str = 'X') -> MaterialProperties:
        """
        计算整体材料属性
        
        参数:
            h: 梁总厚度
            num_layers: 层数
            W_Gr: 石墨烯质量分数
            H_Gr: 石墨烯形状因子
            T: 温度
            distribution_type: 分布类型
            
        返回:
            MaterialProperties对象
        """
        T_ratio = T / self.constants.T_0
        
        # 将质量分数转换为体积分数
        if W_Gr > 0:
            V_Gr_base = W_Gr / (W_Gr + (self.constants.rho_Gr / self.constants.rho_Cu) * (1 - W_Gr))
        else:
            V_Gr_base = 0.0
        
        # 初始化结果存储
        layer_props = {}
        Q11_layer = {}
        Q55_layer = {}
        
        # 用于计算有效属性的累加器
        E_effective = 0.0
        nu_effective = 0.0
        alpha_effective = 0.0
        rho_effective = 0.0
        
        layer_thickness = h / num_layers
        
        # 计算每层属性
        for k in range(1, int(num_layers) + 1):
            E, nu, alpha, rho = self.compute_layer_properties(
                k, num_layers, V_Gr_base, T_ratio, H_Gr, distribution_type
            )
            
            # 计算该层的实际石墨烯体积分数（用于记录）
            if num_layers == 1:
                V_Gr_actual = V_Gr_base
            else:
                if distribution_type.upper() == 'X':
                    V_Gr_actual = 2 * V_Gr_base * abs(2*k - num_layers - 1) / num_layers
                elif distribution_type.upper() == 'O':
                    V_Gr_actual = 2 * V_Gr_base * (1 - abs(2*k - num_layers - 1) / num_layers)
                else:  # 'U'
                    V_Gr_actual = V_Gr_base
            
            # 存储层属性
            layer_props[f"layer_{k}"] = {
                "E": E, "nu": nu, "alpha": alpha, "rho": rho,
                "V_Gr": V_Gr_actual
            }
            
            # 计算刚度系数
            Q11 = E / (1 - nu ** 2)
            Q55 = E / (2 * (1 + nu))
            Q11_layer[f"Q11_{k}"] = Q11
            Q55_layer[f"Q55_{k}"] = Q55
            
            # 计算有效属性（基于厚度加权平均）
            weight = layer_thickness / h
            E_effective += E * weight
            nu_effective += nu * weight
            alpha_effective += alpha * weight
            rho_effective += rho * weight
        
        return MaterialProperties(
            E=E_effective,
            nu=nu_effective, 
            alpha=alpha_effective,
            rho=rho_effective,
            layer_properties=layer_props,
            Q11=Q11_layer,
            Q55=Q55_layer
        )
    
    def compute_stiffness_coefficients(self, material_props: MaterialProperties, 
                                     h: float, num_layers: int) -> Tuple[float, float, float, float]:
        """
        计算刚度系数 A11, B11, D11, A55
        
        参数:
            material_props: 材料属性对象
            h: 总厚度
            num_layers: 层数
            
        返回:
            刚度系数: A11, B11, D11, A55
        """
        delta_z = h / num_layers
        
        # 初始化刚度分量
        A11 = 0.0
        B11 = 0.0
        D11 = 0.0
        A55 = 0.0
        
        # 获取各层的刚度值
        Q11_values = list(material_props.Q11.values())
        Q55_values = list(material_props.Q55.values())
        
        # 剪切修正因子
        kappa = 5.0 / 6.0
        
        # 对每层进行积分
        for k in range(int(num_layers)):
            # 层的边界坐标
            z_k = -h / 2.0 + k * delta_z
            z_k_plus_1 = -h / 2.0 + (k + 1) * delta_z
            
            Q11_k = Q11_values[k]
            Q55_k = Q55_values[k]
            
            # 计算各刚度分量的贡献
            # A11 = ∫ Q11 dz
            A11 += Q11_k * (z_k_plus_1 - z_k)
            
            # B11 = ∫ Q11 * z dz
            B11 += Q11_k * (z_k_plus_1**2 - z_k**2) / 2.0
            
            # D11 = ∫ Q11 * z² dz
            D11 += Q11_k * (z_k_plus_1**3 - z_k**3) / 3.0
            
            # A55 = ∫ κ * Q55 dz
            A55 += kappa * Q55_k * (z_k_plus_1 - z_k)
        
        return A11, B11, D11, A55
    
    def compute_thermal_forces(self, material_props: MaterialProperties, 
                             A11: float, B11: float, delta_T: float,
                             h: float, num_layers: int) -> Tuple[float, float]:
        """
        计算热力和热力矩（与MATLAB方法完全一致）
        
        参数:
            material_props: 材料属性
            A11: 拉伸刚度
            B11: 弯拉耦合刚度
            delta_T: 温度变化
            h: 总厚度
            num_layers: 层数
            
        返回:
            热轴力 N_XT, 热弯矩 M_XT
        """
        # 获取各层的刚度值
        Q11_values = list(material_props.Q11.values())
        
        # 初始化
        A11_MN = 0.0
        B11_MN = 0.0
        N_XT = 0.0
        M_XT = 0.0
        
        delta_z = h / num_layers
        
        # 对每层进行计算（与MATLAB完全一致）
        for k in range(int(num_layers)):
            # 获取该层的热膨胀系数
            layer_key = f"layer_{k+1}"  # 1-based key
            if layer_key in material_props.layer_properties:
                alpha_c_k = material_props.layer_properties[layer_key]['alpha']
            else:
                alpha_c_k = material_props.alpha  # fallback
            
            Q11_k = Q11_values[k]
            
            # 计算层边界
            z_k = -h/2.0 + k * delta_z
            z_k_plus_1 = -h/2.0 + (k + 1) * delta_z
            
            # 计算A11贡献（对应MATLAB第103-105行）
            contribution_A11 = Q11_k * (z_k_plus_1 - z_k)
            A11_MN = A11_MN + contribution_A11
            N_XT = -A11_MN * alpha_c_k * delta_T
            
            # 计算B11贡献（对应MATLAB第108-110行）
            contribution_B11 = Q11_k * (z_k_plus_1**2 - z_k**2) / 2.0
            B11_MN = B11_MN + contribution_B11
            M_XT = -B11_MN * alpha_c_k * delta_T
        
        return N_XT, M_XT
    
    def compute_dimensionless_parameters(self, h: float, L: float, num_layers: int,
                                       W_Gr: float = 0.025, H_Gr: float = 0.8,
                                       T: float = 300.0, distribution_type: str = 'X') -> DimensionlessParameters:
        """
        计算无量纲参数
        
        参数:
            h: 梁厚度
            L: 梁长度
            num_layers: 层数
            W_Gr: 石墨烯质量分数
            H_Gr: 石墨烯形状因子
            T: 温度
            distribution_type: 分布类型
            
        返回:
            无量纲参数对象
        """
        # 1. 计算有石墨烯的材料属性
        props = self.compute_material_properties(h, num_layers, W_Gr, H_Gr, T, distribution_type)
        A11, B11, D11, A55 = self.compute_stiffness_coefficients(props, h, num_layers)
        
        # 2. 计算纯铜（参考材料）的属性
        props_0 = self.compute_material_properties(h, num_layers, 0.0, 0.0, T, distribution_type)
        A11_0, B11_0, D11_0, A55_0 = self.compute_stiffness_coefficients(props_0, h, num_layers)
        
        # 3. 计算热力学参数
        delta_T = T - self.constants.T_0
        N_XT, M_XT = self.compute_thermal_forces(props, A11, B11, delta_T, h, num_layers)
        
        # 4. 计算无量纲参数
        a11 = A11 / A11_0
        a55 = A55 / A11_0
        b11 = B11 / (h * A11_0)
        d11 = D11 / (h**2 * A11_0)
        
        # 长高比
        lambda_val = L / h
        
        # 无量纲热力学参数（与MATLAB完全一致）
        # MATLAB: n_xT = N_XT/A11_0; m_xT = M_XT/(h * A11_0);
        n_xT = N_XT / A11_0  # 除以参考刚度进行无量纲化
        m_xT = M_XT / (h * A11_0)  # 除以h*A11_0进行无量纲化
        
        return DimensionlessParameters(
            a11=a11, b11=b11, d11=d11, a55=a55,
            lambda_val=lambda_val, n_xT=n_xT, m_xT=m_xT
        )
    
    def print_material_summary(self, h: float, L: float, num_layers: int,
                             W_Gr: float = 0.025, H_Gr: float = 0.8,
                             T: float = 300.0, distribution_type: str = 'X'):
        """
        打印材料属性摘要
        
        参数:
            h: 梁厚度
            L: 梁长度  
            num_layers: 层数
            W_Gr: 石墨烯质量分数
            H_Gr: 石墨烯形状因子
            T: 温度
            distribution_type: 分布类型
        """
        print("=" * 60)
        print("材料属性计算摘要")
        print("=" * 60)
        print(f"几何参数: L = {L:.3f} m, h = {h:.3f} m, 层数 = {num_layers}")
        print(f"材料参数: W_Gr = {W_Gr:.3f}, H_Gr = {H_Gr:.3f}, T = {T:.1f} K")
        print(f"分布类型: {distribution_type}")
        print("-" * 60)
        
        # 计算材料属性
        props = self.compute_material_properties(h, num_layers, W_Gr, H_Gr, T, distribution_type)
        A11, B11, D11, A55 = self.compute_stiffness_coefficients(props, h, num_layers)
        
        # 纯铜参考
        props_0 = self.compute_material_properties(h, num_layers, 0.0, 0.0, T, distribution_type)
        A11_0, B11_0, D11_0, A55_0 = self.compute_stiffness_coefficients(props_0, h, num_layers)
        
        # 热力学参数
        delta_T = T - self.constants.T_0
        N_XT, M_XT = self.compute_thermal_forces(props, A11, B11, delta_T, h, num_layers)
        
        print("有效材料属性:")
        print(f"  杨氏模量: E = {props.E/1e9:.2f} GPa")
        print(f"  泊松比: ν = {props.nu:.3f}")
        print(f"  热膨胀系数: α = {props.alpha*1e6:.2f} × 10⁻⁶ /K")
        print(f"  密度: ρ = {props.rho:.0f} kg/m³")
        
        print("\n刚度系数:")
        print(f"  A11 = {A11:.2e}")
        print(f"  B11 = {B11:.2e}")
        print(f"  D11 = {D11:.2e}")
        print(f"  A55 = {A55:.2e}")
        
        print("\n纯铜参考值:")
        print(f"  A11_0 = {A11_0:.2e}")
        print(f"  B11_0 = {B11_0:.2e}")
        print(f"  D11_0 = {D11_0:.2e}")
        print(f"  A55_0 = {A55_0:.2e}")
        
        # 无量纲参数
        dimensionless = self.compute_dimensionless_parameters(h, L, num_layers, W_Gr, H_Gr, T, distribution_type)
        
        print("\n无量纲参数:")
        print(f"  a11 = {dimensionless.a11:.4f}")
        print(f"  b11 = {dimensionless.b11:.4f}")
        print(f"  d11 = {dimensionless.d11:.4f}")
        print(f"  a55 = {dimensionless.a55:.4f}")
        print(f"  λ = L/h = {dimensionless.lambda_val:.2f}")
        print(f"  n_xT = {dimensionless.n_xT:.6f}")
        print(f"  m_xT = {dimensionless.m_xT:.6f}")
        
        if delta_T != 0:
            print(f"\n热力学参数 (ΔT = {delta_T:.1f} K):")
            print(f"  N_XT = {N_XT:.2e}")
            print(f"  M_XT = {M_XT:.2e}")
        
        print("=" * 60)


def create_material_calculator(constants: Optional[MaterialConstants] = None) -> MaterialCalculator:
    """
    创建材料计算器的工厂函数
    
    参数:
        constants: 材料常数
        
    返回:
        材料计算器实例
    """
    return MaterialCalculator(constants)

def compute_material_params_for_solver(h: float, L: float, num_layers: int = 10,
                                     W_Gr: float = 0.025, H_Gr: float = 0.8,
                                     T: float = 300.0, distribution_type: str = 'X',
                                     q: float = -0.08) -> Dict[str, float]:
    """
    为求解器计算材料参数
    
    参数:
        h: 梁厚度 (m)
        L: 梁长度 (m)
        num_layers: 层数
        W_Gr: 石墨烯质量分数
        H_Gr: 石墨烯形状因子
        T: 温度 (K)
        distribution_type: 分布类型
        q: 无量纲分布载荷 (dimensionless)
        
    返回:
        求解器所需的材料参数字典
    """
    calculator = create_material_calculator()
    dimensionless = calculator.compute_dimensionless_parameters(
        h, L, num_layers, W_Gr, H_Gr, T, distribution_type
    )
    
    return {
        'a11': dimensionless.a11,
        'b11': dimensionless.b11,
        'd11': dimensionless.d11,
        'a55': dimensionless.a55,
        'lambda_val': dimensionless.lambda_val,  # 与求解器键名保持一致
        'n_xT': dimensionless.n_xT,
        'm_xT': dimensionless.m_xT,
        'q': q  # 直接使用无量纲载荷
    }

if __name__ == "__main__":
    # 测试材料属性计算模块
    print("测试材料属性计算模块...")
    
    # 创建计算器
    calculator = create_material_calculator()
    
    # 测试参数
    h = 0.1  # 梁厚度
    L = 20 * h  # 梁长度
    num_layers = 10
    W_Gr = 0.025
    H_Gr = 1
    T = 325.0
    q = -0.08
    distribution_type = 'X'
    # 打印材料摘要
    calculator.print_material_summary(h, L, num_layers, W_Gr, H_Gr, T, distribution_type)
    
    # 为求解器计算参数
    solver_params = compute_material_params_for_solver(h, L, num_layers, W_Gr, H_Gr, T, distribution_type, q)
    
    print("\n求解器参数:")
    for key, value in solver_params.items():
        print(f"  {key}: {value:.6f}")
    
    print("\n材料属性计算模块测试完成！")
