#include "DensePointCloudGenerator.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

DensePointCloudGenerator::DensePointCloudGenerator() {}
DensePointCloudGenerator::~DensePointCloudGenerator() {}


void DensePointCloudGenerator::setBaseline(double baseline_m) { baseline_m_ = baseline_m; }


intrinsics generateIntrinsics(std::vector<double> K_vec){
  if (K_vec.size() >= 4)
    {
      intrinsics intrinsics;
      intrinsics.fx = K_vec[0];
      intrinsics.fy = K_vec[1];
      intrinsics.cx = K_vec[2];
      intrinsics.cy = K_vec[3];
      return intrinsics;
    }
  else
    {
      std::cerr << "Error: K vector must have at least 4 elements." << std::endl;
      return intrinsics();
    }
}

cv::Mat generateDistortionCoeffs(std::vector<double> dist_vec){
  if (!dist_vec.empty())
    {
      cv::Mat dist = cv::Mat(1, static_cast<int>(dist_vec.size()), CV_64F);
      for (size_t i = 0; i < dist_vec.size(); ++i)
      {
        dist.at<double>(0, static_cast<int>(i)) = dist_vec[i];
      }
      return dist;
    }
  else
    {
      std::cerr << "Error: Distortion vector is empty." << std::endl;
      return cv::Mat();
    }
}

cv::Mat generateTransformationMatrix(std::vector<std::vector<double>> T_vec){
  if (T_vec.size() >= 4 && T_vec[0].size() >= 4)
    {
      cv::Mat T = cv::Mat(4, 4, CV_64F);
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)      
          T.at<double>(i, j) = T_vec[i][j];
      
      
      return T;
    }
  else
    {
      std::cerr << "Error: Transformation vector must have at least 16 elements." << std::endl;
      return cv::Mat();
    }
}

void DensePointCloudGenerator::initFromYaml(const std::string &yaml_file)
{
  cerr << FBLU("Initializing from:") << " " << yaml_file << "\n"
       << "\n";
  YAML::Node configuration = YAML::LoadFile(yaml_file);

  // Loading camera intrinsics from YAML. The project YAML stores them under
  // "camera_intrinsics.K" and "camera_intrinsics.D" (ROS/OpenCV format), while
  // other configs may still use "distortion_coeffs".
  const YAML::Node rgb = configuration["rgb_intrinsics"];
  if (rgb)
  {
    if (!rgb["intrinsics"] || !(rgb["D"] || rgb["distortion_coeffs"]))
    {
      std::cerr << "Error: RGB intrinsics missing required matrices in YAML." << std::endl;
      return;
    }
    std::vector<double> K_vec = rgb["intrinsics"].as<std::vector<double>>();
    intrinsics_rgb = generateIntrinsics(K_vec);
  
    const YAML::Node dist_node = rgb["D"] ? rgb["D"] : rgb["distortion_coeffs"];
    std::vector<double> dist_vec = dist_node.as<std::vector<double>>();
    dist_coeff_rgb = generateDistortionCoeffs(dist_vec);
  
  }

  const YAML::Node ir1 = configuration["ir1_intrinsics"];
  if (ir1)
  {
    if (!ir1["intrinsics"] || !(ir1["D"] || ir1["distortion_coeffs"]))
    {
      std::cerr << "Error: IR1 intrinsics missing required matrices in YAML." << std::endl;
      return;
    }
    std::vector<double> K_vec = ir1["intrinsics"].as<std::vector<double>>();
    intrinsics_ir1 = generateIntrinsics(K_vec);
  
    const YAML::Node dist_node = ir1["D"] ? ir1["D"] : ir1["distortion_coeffs"];
    std::vector<double> dist_vec = dist_node.as<std::vector<double>>();
    dist_coeff_ir1 = generateDistortionCoeffs(dist_vec);
  
  }
  const YAML::Node ir2 = configuration["ir2_intrinsics"];
  if (ir2)
  {
    if (!ir2["intrinsics"] || !(ir2["D"] || ir2["distortion_coeffs"]) || !ir2["T_cn_ir1"] || !ir2["T_cn_rgb"])
    {
      std::cerr << "Error: IR2 intrinsics missing required matrices in YAML." << std::endl;
      return;
    }
    std::vector<double> K_vec = ir2["intrinsics"].as<std::vector<double>>();
    intrinsics_ir2 = generateIntrinsics(K_vec);
  
    const YAML::Node dist_node = ir2["D"] ? ir2["D"] : ir2["distortion_coeffs"];
    std::vector<double> dist_vec = dist_node.as<std::vector<double>>();
    dist_coeff_ir2 = generateDistortionCoeffs(dist_vec);
  
    T_ir2_ir1 = generateTransformationMatrix(ir2["T_cn_ir1"].as<std::vector<std::vector<double>>>());
    
    std::cout << "IR2 Transformation to RGB:\n" << ir2["T_cn_rgb"] << "\n";
    T_ir2_rgb = generateTransformationMatrix(ir2["T_cn_rgb"].as<std::vector<std::vector<double>>>());

    // Extract baseline from transformation matrix (translation component)
    if (!T_ir2_ir1.empty() && T_ir2_ir1.rows == 4 && T_ir2_ir1.cols == 4)
    {
      double tx = T_ir2_ir1.at<double>(0, 3);
      double ty = T_ir2_ir1.at<double>(1, 3);
      double tz = T_ir2_ir1.at<double>(2, 3);
      baseline_m_ = std::sqrt(tx*tx + ty*ty + tz*tz);
    }
  }

  // Loading other parameters from YAML
  // if (configuration["stereo"]["baseline_m"])
  //   baseline_m_ = configuration["stereo"]["baseline_m"].as<double>();
  if (configuration["depth"]["max_depth"])
    max_depth_ = configuration["depth"]["max_depth"].as<double>();
  if (configuration["depth"]["min_depth"])
    min_depth_ = configuration["depth"]["min_depth"].as<double>();

  input_rgb = configuration["input_rgb"] ? configuration["input_rgb"].as<std::string>() : "/maps/frames-RosarioV2-row2/rgb/1703261894400792837.png";     
  input_irR = configuration["input_irR"] ? configuration["input_irR"].as<std::string>() : "/maps/frames-RosarioV2-row2/infra_left(2)/infra2_1703261894400734901.png";
  input_irL = configuration["input_irL"] ? configuration["input_irL"].as<std::string>() : "/maps/frames-RosarioV2-row2/infraRight(1)/infra1_1703261894400734901.png";
  input_csv = configuration["input_csv"] ? configuration["input_csv"].as<std::string>() : "/maps/frames-RosarioV2-row2/utm_jpg_final.csv";
  input_timestamp_rgb = configuration["input_timestamp_rgb"] ? configuration["input_timestamp_rgb"].as<std::string>() : "1703261894400792837";
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr
DensePointCloudGenerator::generate(const cv::Mat &rgb, const cv::Mat &ir_left, const cv::Mat &ir_right)
{
  CV_Assert(!rgb.empty());
  CV_Assert(!ir_left.empty());

  cv::Mat rgb_img, irL, irR;
  if (rgb.channels() == 3)
    rgb_img = rgb;
  else
    cv::cvtColor(rgb, rgb_img, cv::COLOR_GRAY2BGR);

  if (ir_left.channels() > 1)
    cv::cvtColor(ir_left, irL, cv::COLOR_BGR2GRAY);
  else
    irL = ir_left;

  if (!ir_right.empty())
  {
    if (ir_right.channels() > 1)
      cv::cvtColor(ir_right, irR, cv::COLOR_BGR2GRAY);
    else
      irR = ir_right;
  }

  int width = rgb_img.cols;
  int height = rgb_img.rows;

  auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  cloud->width = static_cast<uint32_t>(width);
  cloud->height = static_cast<uint32_t>(height);
  cloud->is_dense = false;
  cloud->points.resize(width * height);

  const bool has_rgb_size = rgb_img.size() == irL.size();

  cv::Mat RT_camera_world = cv::Mat::zeros(4, 4, CV_64F); // Identity matrix for camera to world transformation
  RT_camera_world.at<double>(0, 2) = 1.0;
  RT_camera_world.at<double>(1, 0) = -1.0; 
  RT_camera_world.at<double>(2, 1) = -1.0;
  RT_camera_world.at<double>(3, 3) = 1.0;


  std::cout << "RGB Intrinsics: fx=" << intrinsics_rgb.fx << ", fy=" << intrinsics_rgb.fy << ", cx=" << intrinsics_rgb.cx << ", cy=" << intrinsics_rgb.cy << "\n";
  std::cout << "MaxDepth=" << max_depth_ << " m, MinDepth=" << min_depth_ << " m\n";
  if (!irR.empty() && irR.size() == irL.size())
  {
    
    std::cout << "IR1 Intrinsics: fx=" << intrinsics_ir1.fx << ", fy=" << intrinsics_ir1.fy << ", cx=" << intrinsics_ir1.cx << ", cy=" << intrinsics_ir1.cy << "\n";
    std::cout << "IR2 Intrinsics: fx=" << intrinsics_ir2.fx << ", fy=" << intrinsics_ir2.fy << ", cx=" << intrinsics_ir2.cx << ", cy=" << intrinsics_ir2.cy << "\n";
    std::cout << "IR baseline: " << baseline_m_ << " m\n";

    // Stereo matching on IR pair.
    int numDisparities = ((width / 8) + 15) & -16;
    int blockSize = 7;
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, numDisparities, blockSize);
    sgbm->setP1(8 * blockSize * blockSize);
    sgbm->setP2(32 * blockSize * blockSize);
    sgbm->setMode(cv::StereoSGBM::MODE_SGBM);

    cv::Mat disp16s, disp;
    sgbm->compute(irL, irR, disp16s);
    disp16s.convertTo(disp, CV_32F, 1.0 / 16.0);
    // disp.normalize(0, 255, cv::NORM_MINMAX, CV_8U);
    // cv::Mat depth_map = intrinsics_ir2.fx * baseline_m_ / disp; // Depth map in meters
    // cv::imwrite(getPackagePath() + "/params/output/clouds/" + input_timestamp_rgb + "_depth_map.png", depth_map);
    // Reconstruct depth in the left IR camera frame using the left IR intrinsics
    // and the baseline between the two IR cameras.
    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {        
        const float d = disp.at<float>(y, x);
        pcl::PointXYZRGB &pt = cloud->at(x, y);
        if (d > 0.0f ) //&& x >= static_cast<int>(width * 0.2) && x < static_cast<int>(width * 0.8)
        {
          const double Z = intrinsics_ir2.fx * baseline_m_ / d;
          if (Z > min_depth_ && Z < max_depth_)
          {
            // std::cout << "Disparity at (" << x << ", " << y << "): " << d << ", Depth: " << Z << "\n";
            const double X = (x - intrinsics_ir2.cx) * Z / intrinsics_ir2.fx;
            const double Y = (y - intrinsics_ir2.cy) * Z / intrinsics_ir2.fy;
            cv::Mat_<double> pt_ir2 = (cv::Mat_<double>(4, 1) << X, Y, Z,0.0);

            // 3. Transformar el punto 3D del sistema IR2 al sistema RGB
            // P_rgb = R * P_ir2 + T
            cv::Mat pt_rgb_mat = T_ir2_rgb * pt_ir2;          
            
            const double pt_x = pt_rgb_mat.at<double>(0, 0);
            const double pt_y = pt_rgb_mat.at<double>(1, 0);
            const double pt_z = pt_rgb_mat.at<double>(2, 0);
            
            const int rgb_x = static_cast<int>(std::round((intrinsics_rgb.fx * pt_x / pt_z) + intrinsics_rgb.cx));
            const int rgb_y = static_cast<int>(std::round((intrinsics_rgb.fy * pt_y / pt_z) + intrinsics_rgb.cy));
            if (rgb_x >= 0 && rgb_x < width && rgb_y >= 0 && rgb_y < height)
            {
              cv::Mat real_pt_rgb_mat = RT_camera_world * pt_rgb_mat;              
              pt.x = static_cast<float>(real_pt_rgb_mat.at<double>(0, 0));
              pt.y = static_cast<float>(real_pt_rgb_mat.at<double>(1, 0));
              pt.z = static_cast<float>(real_pt_rgb_mat.at<double>(2, 0));

              cv::Vec3b color = rgb_img.at<cv::Vec3b>(rgb_y, rgb_x);
              pt.r = color[2];
              pt.g = color[1];
              pt.b = color[0];
              continue;
            }
          }
        }
        pt.x = pt.y = pt.z = std::numeric_limits<float>::quiet_NaN();
        pt.r = pt.g = pt.b = 0;
      }
    }    
  }
  // else
  // {
  //   // Single IR image heuristic. Use IR1 intrinsics when available, otherwise
  //   // fall back to RGB intrinsics.
  //   const double ir_fx = (K_ir1.empty() ? rgb_fx : K_ir1.at<double>(0, 0));
  //   const double ir_fy = (K_ir1.empty() ? rgb_fy : K_ir1.at<double>(1, 1));
  //   const double ir_cx = (K_ir1.empty() ? rgb_cx : K_ir1.at<double>(0, 2));
  //   const double ir_cy = (K_ir1.empty() ? rgb_cy : K_ir1.at<double>(1, 2));
  //   std::cout << "Single IR Intrinsics: fx=" << ir_fx << ", fy=" << ir_fy << ", cx=" << ir_cx << ", cy=" << ir_cy << "\n";

  //   cv::Mat irf;
  //   irL.convertTo(irf, CV_32F, 1.0 / 255.0);
  //   for (int y = 0; y < height; ++y)
  //   {
  //     for (int x = 0; x < width; ++x)
  //     {
  //       const float intensity = irf.at<float>(y, x);
  //       const float Z = (1.0f - intensity) * max_depth_;
  //       pcl::PointXYZRGB &pt = cloud->at(x, y);
  //       if (Z > 0.0f && Z < max_depth_)
  //       {
  //         const double X = (x - ir_cx) * Z / ir_fx;
  //         const double Y = (y - ir_cy) * Z / ir_fy;
  //         pt.x = static_cast<float>(X);
  //         pt.y = static_cast<float>(Y);
  //         pt.z = static_cast<float>(Z);
  //         cv::Vec3b color = rgb_img.at<cv::Vec3b>(y, x);
  //         pt.r = color[2];
  //         pt.g = color[1];
  //         pt.b = color[0];
  //       }
  //       else
  //       {
  //         pt.x = pt.y = pt.z = std::numeric_limits<float>::quiet_NaN();
  //         pt.r = pt.g = pt.b = 0;
  //       }
  //     }
  //   }
  // }
  
  return cloud;
}


/*
Para generar una nube de puntos densa y coloreada utilizando la información de profundidad de las cámaras infrarrojas (IR) y el color de la cámara RGB, debes seguir un pipeline de procesamiento de visión por computadora.

Dado que tienes las matrices de rototraslación (transformación extrínseca) y, presumiblemente, los **parámetros intrínsecos** de cada cámara (matriz de calibración con distancia focal y centro óptico), el procedimiento estándar se divide en los siguientes pasos:

---

### 1. Generación de la nube de puntos desde las cámaras IR

Primero, debes decidir si vas a usar **una sola cámara IR** para la profundidad (por ejemplo, `ir1`) o si tienes un sistema estéreo IR (`ir1` + `ir2`) para calcular un mapa de disparidad/profundidad más denso.

* **Si usas un mapa de disparidad estéreo (IR1 - IR2):**
1. Alinea las imágenes IR usando la matriz de rototraslación entre `ir2` e `ir1` (rectificación estéreo).
2. Calcula el **mapa de disparidad** (usando algoritmos como SGM o BM).
3. Convierte el mapa de disparidad a un **mapa de profundidad** ($Z$) usando la línea base de las cámaras y la distancia focal.
4. Utiliza los parámetros intrínsecos de `ir1` para hacer el "back-projection" (retroproyección) de cada píxel $(u, v)$ del mapa de profundidad a coordenadas tridimensionales $(X, Y, Z)$ en el sistema de referencia de `ir1`.


* **Fórmula de retroproyección (para cada píxel IR):**

$$X = \frac{(u - c_x) \cdot Z}{f_x}$$


$$Y = \frac{(v - c_y) \cdot Z}{f_y}$$


$$Z = Z$$



*(Donde $f_x, f_y, c_x, c_y$ son los intrínsecos de la cámara IR).*

---

### 2. Transformación al sistema de coordenadas de la cámara RGB

Como la nube de puntos inicial está en el sistema de coordenadas de `ir1` (o `ir2`), necesitas llevar esos puntos 3D al sistema de coordenadas de la **cámara RGB**.

Para esto utilizas la matriz de rototraslación que relaciona la cámara IR con la RGB ($R_{ir \to rgb}$ y $T_{ir \to rgb}$).

* *Nota:* Si tienes la matriz de `ir2` a `rgb` y de `ir2` a `ir1`, puedes concatenar/invertir las matrices para obtener la transformación directa de la cámara IR que usaste para la profundidad hacia la cámara RGB.

Para cada punto 3D $P_{ir} = (X_{ir}, Y_{ir}, Z_{ir})^T$:


$$P_{rgb} = R_{ir \to rgb} \cdot P_{ir} + T_{ir \to rgb}$$

---

### 3. Proyección y Mapeo de Color (Texturizado)

Una vez que tienes los puntos 3D expresados en el sistema de coordenadas de la cámara RGB, puedes proyectarlos sobre el plano de la imagen RGB para extraer su color correspondiente.

1. **Proyección en el plano de la imagen RGB:**
Utiliza los parámetros intrínsecos de la cámara RGB ($f_{x,rgb}, f_{y,rgb}, c_{x,rgb}, c_{y,rgb}$) para proyectar el punto $P_{rgb} = (X_{rgb}, Y_{rgb}, Z_{rgb})^T$ a coordenadas de píxel $(u_{rgb}, v_{rgb})$:

$$u_{rgb} = f_{x,rgb} \cdot \frac{X_{rgb}}{Z_{rgb}} + c_{x,rgb}$$


$$v_{rgb} = f_{y,rgb} \cdot \frac{Y_{rgb}}{Z_{rgb}} + c_{y,rgb}$$


2. **Asignación de Color:**
* Verifica que $(u_{rgb}, v_{rgb})$ caigan dentro de los límites de la resolución de la imagen RGB.
* Extrae el valor de color (RGB) de ese píxel en la imagen.
* Asigna ese color al punto 3D correspondiente en la nube de puntos.



---

### 4. Filtrado y Limpieza recomendada

Al combinar múltiples fuentes de cámaras, es muy común encontrar ruido o artefactos. Te sugiero aplicar los siguientes filtros en tu nube de puntos final:

* **Filtrado por Z (Depth Threshold):** Elimina puntos con profundidad cero, negativa o extremadamente lejana donde el sensor IR pierde precisión.
* **Oclusión (Z-buffer check):** Si un punto se proyecta fuera de los límites de la imagen RGB o queda oculto por otro objeto, descártalo o asígnale un color nulo.
* **Filtro Estadístico de Outliers (SOR):** Útil para eliminar puntos flotantes en el espacio generados por ruido infrarrojo.

---

### Herramientas recomendadas para la implementación

Para programar esto de forma eficiente sin reinventar la rueda, puedes utilizar:

* **OpenCV:** Para el manejo de imágenes, rectificación, operaciones matriciales y proyección perspectiva (`cv2.projectPoints`).
* **Open3D** o **PCL (Point Cloud Library):** Ideales para almacenar la estructura de la nube de puntos, aplicar transformaciones rígidas con matrices de rototraslación y realizar filtrados o visualización 3D interactiva.
*/