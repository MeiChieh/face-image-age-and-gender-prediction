import cv2
from PIL import Image
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from helper import *
from sklearn.cluster import KMeans
from typing import Tuple, List, Dict
import os

def detect_light_anamolies(
    full_df: pd.DataFrame, 
    low_threshold: int, 
    high_threshold: int, 
    pixels_percentage: float
) -> Tuple[List[str], List[str]]:
    """
    Detects images with low or high light anomalies based on specified thresholds.

    Args:
        full_df (pd.DataFrame): DataFrame containing image file names in a column named 'file_name'.
        low_threshold (int): Pixel intensity threshold for detecting low light.
        high_threshold (int): Pixel intensity threshold for detecting high light.
        pixels_percentage (float): Percentage of pixels that need to be below/above the thresholds 
                                   to classify the image as low/high light.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists:
            - List of file names classified as low light.
            - List of file names classified as high light.
    """
    low_light_ls = []
    high_light_ls = []

    file_name_ls = full_df.file_name.tolist()

    for file_name in file_name_ls:
        with Image.open(file_name) as img:
            img = img.convert("L")
            img_arr = np.array(img)

            is_low_light = (img_arr < low_threshold).mean() > (pixels_percentage / 100)

            is_high_light = (img_arr > high_threshold).mean() > (
                pixels_percentage / 100
            )

            if is_low_light:
                low_light_ls.append(file_name)
            if is_high_light:
                high_light_ls.append(file_name)

    return (low_light_ls, high_light_ls)


def plot_img_ls(
    img_ls: List[str], 
    filename_id_dict: Dict[str, str], 
    s1: int, 
    s2: int, 
    suptitle: str, 
    suptitle_y: float = 1
) -> None:
    """
    Plots a list of images in a grid format with titles and a super title.

    Args:
        img_ls (List[str]): List of file names of images to be plotted.
        filename_id_dict (Dict[str, str]): Dictionary mapping file names to their respective titles.
        s1 (int): Number of rows in the subplot grid.
        s2 (int): Number of columns in the subplot grid.
        suptitle (str): Title for the entire figure.
        suptitle_y (float, optional): Y-position for the super title. Default is 1.

    Returns:
        None
    """

    n = 1
    for file_name in img_ls:
        plt.subplot(s1, s2, n)
        plt.imshow(Image.open(file_name))
        plt.axis("off")

        n += 1
        plt.title(filename_id_dict[file_name])

    plt.suptitle(suptitle, y=suptitle_y)
    plt.show()

def plot_duplicated_imgs(
    duplicated_img_df: pd.DataFrame, 
    slice_range_ls: Tuple[int, int], 
    plot_filename: str
) -> None:
    """
    Plots duplicated images from a DataFrame and saves the figure if it doesn't already exist.

    Args:
        duplicated_img_df (pd.DataFrame): DataFrame containing file names of duplicated images.
        slice_range_ls (Tuple[int, int]): Range of indices to slice the DataFrame for plotting.
        plot_filename (str): Filename for saving the plotted figure.

    Returns:
        None
    """
    if os.path.exists(plot_filename):
        img = Image.open(plot_filename)
        plt.figure(figsize=(20, 20))
        plt.imshow(img)
        plt.axis("off")
        plt.show()
        return

    n = 0
    plt.figure(figsize=(20, 20))

    filename_hash_dict = {
        v: k for k, v in duplicated_img_df.file_name.to_dict().items()
    }

    dup_imgs_subset = tuple(
        duplicated_img_df.file_name.tolist()[slice_range_ls[0] : slice_range_ls[1]]
    )

    for i in dup_imgs_subset:
        plt.subplot(20, 20, n + 1)
        plt.imshow(Image.open(i))
        plt.title(filename_hash_dict[i])
        plt.axis("off")
        n += 1
    plt.subplots_adjust(hspace=1)
    plt.suptitle(
        f"Duplicated Images from {slice_range_ls[0]} to {slice_range_ls[1]}", y=0.91
    )
    plt.savefig(plot_filename, bbox_inches="tight")
    plt.show()

def construct_pred_res_df(
    pred_res: List[np.ndarray], test_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Constructs a DataFrame with predicted and true labels for age and gender.

    Args:
        pred_res (List[np.ndarray]): List of predictions, where each element contains predicted gender and age probabilities.
        test_df (pd.DataFrame): DataFrame containing the test dataset with true labels.

    Returns:
        pd.DataFrame: A DataFrame containing the original test data along with predictions and true labels for age and gender.
    """
    # Get prediction labels
    gender_last, age_last = list(pred_res[-1][1]), list(pred_res[-1][0])
    gender_f_prob_pred = [
        i[0] for i in np.array(pred_res[:-1])[:, 1, :].reshape(-1, 1)
    ] + gender_last
    age_pred = [
        i[0] for i in np.array(pred_res[:-1])[:, 0, :].reshape(-1, 1)
    ] + age_last
    gender_pred = [1 if i > 0.5 else 0 for i in gender_f_prob_pred]

    # get true labels
    age_true = test_df.label_tensor.apply(lambda i: i[0].item())
    gender_true = test_df.label_tensor.apply(lambda i: i[1].item())

    res_df = test_df.copy()
    res_df["gender_f_prob_pred"] = gender_f_prob_pred
    res_df["age_pred"] = age_pred
    res_df["gender_pred"] = gender_pred
    res_df["age_true"] = age_true
    res_df["gender_true"] = gender_true
    res_df["gender_tp"] = (res_df.gender_pred == 1) & (res_df.gender_true > 0)
    res_df["gender_tn"] = (res_df.gender_pred == 0) & (res_df.gender_true < 1)
    res_df["gender_fp"] = (res_df.gender_pred == 1) & (res_df.gender_true < 1)
    res_df["gender_fn"] = (res_df.gender_pred == 0) & (res_df.gender_true > 0)
    res_df["age_group"] = pd.cut(res_df.age_true, [0, 5, 13, 25, 40, 60, 80, 120])

    race_filename_ls = res_df.file_name.tolist()
    race_ls = [i.split("/")[3] for i in race_filename_ls]
    race_ls = [i.split("_")[2] for i in race_ls]
    res_df["race"] = race_ls
    race_map = {
        "0": "white",
        "1": "black",
        "2": "asian",
        "3": "indian",
        "4": "hispanic",
    }
    res_df["race_name"] = res_df.race.apply(lambda i: race_map[i])

    res_df["practical_age_group"] = pd.cut(
        res_df.age_true,
        bins=[0, 3, 7, 14, 18, 26, 41, 66, 81, 120],
        labels=[
            "baby (0~3)",
            "pre_school (4~6)",
            "pre-teens (7~13)",
            "teenager (14~17)",
            "young_adult (18~25)",
            "adult (26~40)",
            "middle_age (41~65)",
            "elderly (66~80)",
            "very_elderly (81~120)",
        ],
    )
    res_df["age_pred_neg"] = res_df.age_pred.apply(lambda i: 0 if i < 0 else i)
    res_df["age_diff"] = res_df["age_pred"] - res_df["age_true"]
    res_practical_age_group_desc = (
        res_df.groupby("practical_age_group")["age_diff"]
        .describe()
        .sort_index()
        .round(2)
    )
    age_correction_map = res_practical_age_group_desc["50%"].to_dict()

    res_df["age_pred_median_corrected"] = (
        res_df["practical_age_group"]
        .apply(lambda i: age_correction_map[i])
        .astype("float64")
    )
    res_df["age_pred_median_corrected"] = (
        res_df.age_pred - res_df.age_pred_median_corrected
    )
    # make sure there's no age < 0
    res_df["age_pred_median_corrected"] = res_df.age_pred_median_corrected.apply(
        lambda i: 0 if i < 0 else i
    )
    res_df["age_diff_median_corrected"] = (
        res_df["age_pred_median_corrected"] - res_df["age_true"]
    )

    return res_df
    
def plot_pred_res_images(
    pred_res_group_df: pd.DataFrame, 
    group_name: str
) -> None:
    """
    Plots images from a DataFrame of predicted results.

    Args:
        pred_res_group_df (pd.DataFrame): DataFrame containing the file names of the images to be plotted.
        group_name (str): Title for the plot, indicating the group of images.

    Returns:
        None: Displays a plot of images with their indices.
    """
    plt.figure(figsize=(18, 18))
    if pred_res_group_df.shape[0] < 81:
        sample_num =  pred_res_group_df.shape[0]
    else:
        sample_num = 81

    for idx, i in enumerate(pred_res_group_df.iloc[:sample_num, :].file_name.tolist()):
        with Image.open(i) as img:
            plt.subplot(9, 9, idx + 1)
            plt.imshow(img)
            plt.title(idx)
            plt.axis('off')
    plt.suptitle(f'{group_name} in Gender Prediction', y=0.91)
    plt.show()
    
def detect_saturation_and_brightness(image_path: str) -> Tuple[float, float]:
    """
    Detects the median saturation and brightness of an image.

    Args:
        image_path (str): The file path to the image.

    Returns:
        Tuple[float, float]: A tuple containing the median saturation and median brightness 
                             of the image, both normalized to the range [0, 1].
    """
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation_channel = hsv_image[:, :, 1]
    brightness_channel = hsv_image[:, :, 2]

    saturation_channel_normalized = saturation_channel / 255.0
    brightness_channel_normalized = brightness_channel / 255.0

    median_saturation = np.median(saturation_channel_normalized)
    median_brightness = np.median(brightness_channel_normalized)

    return (median_saturation, median_brightness)

def plot_saturation_and_brightness(
    sample_df: pd.DataFrame, cat_ls: List[str], feature: str, feature_name: str
) -> None:
    """
    Plots the median saturation and brightness for different categories in the dataset.

    Args:
        sample_df (pd.DataFrame): DataFrame containing image file names and category features.
        cat_ls (List[str]): List of category strings to filter the DataFrame.
        feature (str): Column name in `sample_df` containing the category information.
        feature_name (str): Name of the feature to display in the plot title.

    Returns:
        None: This function does not return a value; it displays a plot.
    """

    fig_size(len(cat_ls) * 2, 2)
    hsv_dict = {}
    for idx, i in enumerate(cat_ls):
        cat_filename = sample_df.loc[sample_df[feature] == i].file_name.tolist()

        saturation_ls = []
        brightnes_ls = []
        for j in cat_filename:
            saturation_median, brightness_median = detect_saturation_and_brightness(j)

            saturation_ls.append(saturation_median)
            brightnes_ls.append(brightness_median)

        hsv_dict[i] = (np.median(saturation_ls), np.median(brightnes_ls))

    hsv_df = (
        pd.DataFrame.from_dict(
            hsv_dict,
        )
        .T.rename(columns={0: "saturation", 1: "brightness"})
        .T
    )
    plt.subplot(1, 2, 1)
    sns.barplot(hsv_df.iloc[0, :], palette="coolwarm")
    plt.xticks(rotation=45, ha="right")
    plt.title("Saturation")
    plt.subplot(1, 2, 2)
    sns.barplot(hsv_df.iloc[1, :], palette="coolwarm")
    plt.xticks(rotation=45, ha="right")
    plt.title("Brightness")
    plt.ylabel("")
    plt.suptitle(f"Brightness and Saturation Across {feature_name}", y=1.1)
    plt.show()
    
def get_major_img_colors(
    color_df: pd.DataFrame, 
    feature: str, 
    feat_name_ls: List[str]
) -> Dict[str, np.ndarray]:
    """
    Extracts the major colors from images based on the specified feature.

    Args:
        color_df (pd.DataFrame): DataFrame containing image file names and their associated features.
        feature (str): The column name in `color_df` used for filtering images.
        feat_name_ls (List[str]): List of feature names to analyze.

    Returns:
        Dict[str, np.ndarray]: A dictionary where keys are feature names and values are arrays of major colors extracted from the images.
    """
    
    maj_colors_dict: Dict[str, np.ndarray] = {}
    kmeans = KMeans(n_clusters=15, random_state=0)

    for feat_name in feat_name_ls:
        for i in color_df.loc[color_df[feature] == feat_name].sample(200).file_name:
            with Image.open(i) as img:
                img_arr = np.array(img)
                img_arr = img_arr.reshape(-1, 3)

                max_5_colors = kmeans.fit(img_arr).cluster_centers_

                if feat_name in maj_colors_dict:
                    maj_colors_dict[feat_name] = np.vstack(
                        (maj_colors_dict[feat_name], max_5_colors)
                    )
                else:
                    maj_colors_dict[feat_name] = np.array(max_5_colors)
    return maj_colors_dict


def plot_color_palette(
    major_colors_dict: Dict[str, List[np.ndarray]], 
    feature_name: str
) -> None:
    """
    Plots the major color palette for each feature based on KMeans clustering.

    Args:
        major_colors_dict (Dict[str, List[np.ndarray]]): A dictionary where keys are feature names and 
        values are arrays of major colors extracted from the images.
        feature_name (str): The name of the feature for the plot title.

    Returns:
        None: The function displays the plot but does not return a value.
    """
    
    summerize_kmeans = KMeans(n_clusters=10, random_state=0)
    color_dict = {}

    for key in major_colors_dict:
        summerize_kmeans.fit(major_colors_dict[key])

        pie_color = summerize_kmeans.cluster_centers_ / 255
        label_counts = np.bincount(summerize_kmeans.labels_)

        percentages = label_counts / np.sum(label_counts)

        color_dict[key] = [pie_color, percentages]

    n = len(color_dict)
    fig_size(n * 2, 3)
    for idx, key in enumerate(color_dict):
        pie_color, percentages = color_dict[key]

        plt.subplot(1, n, idx + 1)
        plt.pie(
            percentages,
            colors=pie_color,
        )
        plt.title(key)

    plt.tight_layout()
    plt.suptitle(f"Major Color Palette for {feature_name}")
    plt.show()
    
def edge_detection(img_path: str) -> np.ndarray:
    """
    Performs edge detection on an image using the Canny edge detection method.

    Args:
        img_path (str): The file path of the image to process.

    Returns:
        np.ndarray: An array representing the edges detected in the image.
    """
    # Read the image from the specified file path
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges


def plot_edge_detection(sample_df: pd.DataFrame, cat_ls: List[str], feature: str, feature_name: str) -> None:
    """
    Plots the cumulative edge detection results for specified categories in a dataset.

    Args:
        sample_df (pd.DataFrame): DataFrame containing image file paths and feature information.
        cat_ls (List[str]): List of categories to analyze.
        feature (str): The column name in the DataFrame that contains the feature of interest.
        feature_name (str): The name of the feature, used for the plot title.

    Returns:
        None: This function displays the plot and does not return any value.
    """
    fig_size(len(cat_ls) * 2, 2)
    for idx, i in enumerate(cat_ls):
        cat_filename = sample_df.loc[
            sample_df[feature] == i
        ].file_name.tolist()

        plt.subplot(1, len(cat_ls), idx + 1)
        base_edge = None
        for j in cat_filename:
            # for better visualization, invert the pixel values
            edge = 255 - edge_detection(j)

            if base_edge is None:
                base_edge = edge
            else:
                base_edge += edge

        plt.imshow(base_edge, cmap="vlag")
        plt.axis("off")
        plt.title(i)

        plt.suptitle(f"{feature_name} Edge Detection", y=1.1)
        
def get_res_outliers(res_df):
    quantiles = (
        res_df.groupby("practical_age_group")["age_diff_median_corrected"]
        .quantile([0.01, 0.99])
        .unstack()
    )

    quantiles.columns = ["1st_percentile", "99th_percentile"]
    merged_df = res_df.merge(quantiles, on="practical_age_group")
    below_1st = merged_df[
        merged_df["age_diff_median_corrected"] < merged_df["1st_percentile"]
    ]
    above_99th = merged_df[
        merged_df["age_diff_median_corrected"] > merged_df["99th_percentile"]
    ]
    outliers_df = pd.concat([below_1st, above_99th])
    return outliers_df


def show_race_impact_in_pred_errors(
    res_df: pd.DataFrame, error: str, mark_id: str
) -> None:
    """
    Analyze and display the impact of race on prediction errors.

    Args:
        res_df (pd.DataFrame): DataFrame containing prediction results with race and error information.
        error (str): The type of error to analyze (e.g., 'gender').
        mark_id (str): Identifier for coloring the DataFrame display.

    Returns:
        None: Displays a styled DataFrame with race impact on prediction errors.
    """
    error_race_df = (
        res_df.query(f"gender_{error} == True")
        .groupby("race_name")["age_diff"]
        .agg([("count", "count")])
        .reset_index()
    )
    total_race_proportion = (
        (res_df.groupby("race_name")["age_diff"].count() / res_df.shape[0])
        .round(2)
        .tolist()
    )
    error_race_df[f"{error}_proportion"] = (
        error_race_df["count"] / sum(error_race_df["count"])
    ).round(2)
    error_race_df["race_proportion"] = total_race_proportion
    
    return error_race_df

    # dp(error_race_df.style.apply(mark_df_color, id=mark_id, color="lightgreen"))