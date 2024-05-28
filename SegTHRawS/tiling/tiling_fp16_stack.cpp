#include <iostream>
#include <vector>
#include <fstream> 

// COnstants for image and patch size
// COnstants for image and patch size
// const int full_image_height = 1152;
// const int full_image_width = 1296;
// const int Channels = 3;

// const int patch_height = 256;
// const int patch_width = 256;

const int full_image_height = 256;
const int full_image_width = 256;
const int Channels = 3;

const int patch_height = 64;
const int patch_width = 64;

typedef _Float16 float16;

// Function to read the binary image conatining float16 values
std::vector<float16> read_float16_file(const std::string& filename) {
    
    // Open the binary image
    FILE* pFile = fopen(filename.c_str(), "rb");
    if (pFile == nullptr) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    // Get the file size
    fseek(pFile, 0, SEEK_END);
    long lSize = ftell(pFile);
    rewind(pFile);

    // Allocate memory /*  */to contain the binary image
    std::vector<float16> buffer(lSize / sizeof(float16));

    // Read the image into the buffer
    size_t result = fread(buffer.data(), sizeof(float16), buffer.size(), pFile);
    if (result != buffer.size()) {
        perror("Error reading file");
        exit(EXIT_FAILURE);
    }

    // Close the file
    fclose(pFile);

    return buffer;
}

void save_patches_to_binary_file(const std::vector<std::vector<std::vector<std::vector<float16>>>>& patches, const std::string& filename) {
    
    // Open the new file for writing
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Failed to open file for writing" << std::endl;
        return;
    }

    // Iterate over each patch and write its data to the output file
    for (const auto& patch : patches) {
        for (const auto& row : patch) {
            for (const auto& pixel : row) {
                // Write each pixel value to the file
                file.write(reinterpret_cast<const char*>(pixel.data()), pixel.size() * sizeof(float16));
            }
        }
    }

    // Close the file
    file.close();

    std::cout << "Patches saved to binary file: " << filename << std::endl;
}

#include <chrono>

int main(int argc, char* argv[]) {

    auto start = std::chrono:: high_resolution_clock::now();

    std::string image_shape_type;

    if (argc < 2) {
        image_shape_type = "interlinear";
    }
    else{
        image_shape_type = argv[1];

    }

    std:: cout << "Selected image shape type: " << image_shape_type << std:: endl;

    // File path of the binary image
    std::string filename = "image_fp16.bin";

    // Read the image values
    std::vector<float16> float_buffer = read_float16_file(filename);

    // Create a 3D array to store the final image
    float16 final_image[256][256][3];

    // Define loop indexes to fill the 3D array
    int height_idx = 0;
    int width_idx = 0;
    int channel_idx = 0;

    for (int idx = 0; idx < float_buffer.size(); ++idx) {

        final_image[height_idx][width_idx][channel_idx] = float_buffer[idx];

        // Update indices
        ++channel_idx;
        if (channel_idx == Channels) {
            channel_idx = 0;
            ++width_idx;
            if (width_idx == full_image_width) {
                width_idx = 0;
                ++height_idx;
            }
        }
        if (height_idx == full_image_height) {
            break; // Break loop when the enitre image has been extracted
        }
    }

    // Number of patches in each dimension
    int numPatchesHeight = full_image_height / patch_height;
    int numPatchesWidth = full_image_width / patch_width;

    // Vector to store extracted patches
    std::vector<std::vector<std::vector<std::vector<float16>>>> patches;

    // Extract patches from the original array
    for (int patchRow = 0; patchRow < numPatchesHeight; ++patchRow) {
        for (int patchCol = 0; patchCol < numPatchesWidth; ++patchCol) {

            std::vector<std::vector<std::vector<float16>>> patch(patch_height,
                                                                std::vector<std::vector<float16>>(patch_width,
                                                                                                std::vector<float16>(Channels)));

            // Copy values from the original array to the patch
            if (image_shape_type == "interlinear"){
                // Interlinear format (height, width, channels)
                for (int i = 0; i < patch_height; ++i) {
                    for (int j = 0; j < patch_width; ++j) {
                        for (int k = 0; k < Channels; ++k) {
                            patch[i][j][k] = final_image[patchRow * patch_height + i][patchCol * patch_width + j][k];
                            // std :: cout << patch[i][j][k] << " ";
                        }
                        // std::cout << std:: endl;
                    }
                    // std::cout << std:: endl;
                }
            }
            else if (image_shape_type == "planar"){
                // Planar format (channels, height, width)
                for (int k = 0; k < Channels; ++k) {
                    for (int i = 0; i < patch_height; ++i) {
                        for (int j = 0; j < patch_width; ++j) {
                            patch[i][j][k] = final_image[patchRow * patch_height + i][patchCol * patch_width + j][k];
                            // std :: cout << patch[i][j][k] << " ";
                        }
                        // std::cout << std:: endl;
                    }
                    // std::cout << std:: endl;
                }
            }
            else {
                std::cout << "Error: Image shape type not supported";
            }
            // Store the extracted patch
            patches.push_back(patch);
        }
    }

    std::string outputFilename = "patches_fp16.bin";
    save_patches_to_binary_file(patches, outputFilename);


    auto stop = std::chrono:: high_resolution_clock::now();

    auto duration =std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    
    // To get the value of duration use the count()
    // member function on the duration object
    std:: cout << duration.count() << std:: endl;
    // Display the number of extracted patches
    std::cout << "Number of patches: " << patches.size() << std::endl;

    return 0;
}
