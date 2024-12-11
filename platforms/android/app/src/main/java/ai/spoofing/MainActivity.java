// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package ai.spoofing;
import ai.spoofing.databinding.ActivityMainBinding;
import ai.spoofing.*;

import android.Manifest;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.MediaMetadataRetriever;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;

import android.view.Surface;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;


import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AlertDialog;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.*;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import kotlinx.coroutines.*;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.ArrayList;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.InputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;



import android.widget.TextView;
//import androidx.camera.view.PreviewView;
import android.widget.MediaController;
import android.widget.VideoView;

import com.google.common.util.concurrent.ListenableFuture;

import org.opencv.core.Core;

class Result {
    public List<Integer> detectedIndices;
    public List<Float> detectedScore;
    public long processTimeMs;

    public Result(List<Integer> detectedIndices, List<Float> detectedScore, long processTimeMs) {
        this.detectedIndices = detectedIndices;
        this.detectedScore = detectedScore;
        this.processTimeMs = processTimeMs;
    }
}

public class MainActivity extends AppCompatActivity {
    ActivityMainBinding binding;

    private ExecutorService backgroundExecutor = Executors.newSingleThreadExecutor();
    private List<String> labelData;

    private static final String TAG = "ORTSpoofing";

    private static final double SCORE_SPOOF = 0.6345;

    static {
        System.loadLibrary("spoofing");
        System.loadLibrary("ortcxx");
        System.loadLibrary("onnxruntime");
    };

    private Button uploadButton;
    private View myview;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot()); // Adjust layout reference if needed

        // Init model
        try {
            labelData = readLabels();
            TensorUtils.initModel(getAssets(), readYolo(), readExtractor(), readEmbedder());
        } catch (Exception e) {
            Log.e(TAG, "Error initializing models", e);
        }

        uploadButton = findViewById(R.id.button2);
        uploadButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Show a dialog or list to choose from assets
                showAssetPicker();
            }
        });
    }

    private void showAssetPicker() {
        // Get a list of asset filenames
        String[] assetFiles = null;
        try {
            assetFiles = getAssets().list("tests/"); // List all files in assets folder
            for (String file : assetFiles) {
                Log.d("Assets", file);
            }
        } catch (IOException e) {
            Log.e(TAG, "Error listing assets", e);
            return;
        }

        // Filter for images and videos
        List<String> imageVideoFiles = new ArrayList<>();
        for (String file : assetFiles) {
            if (file.endsWith(".jpg") || file.endsWith(".png") || file.endsWith(".mp4")) {
                imageVideoFiles.add(file);
            }
        }

        // Create and show the dialog
        new AlertDialog.Builder(this)
                .setTitle("Choose an Asset")
                .setItems(imageVideoFiles.toArray(new String[0]), new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        String selectedAsset = imageVideoFiles.get(which);
                        // Handle the selected asset
                        handleSelectedAsset(selectedAsset);
                    }
                })
                .show();
    }

    private void handleSelectedAsset(String assetFilename) {
        try {
            if (assetFilename.endsWith(".jpg") || assetFilename.endsWith(".png")) {
                //set video view to gone
                findViewById(R.id.MainVideoView).setVisibility(View.GONE);
                findViewById(R.id.MainImageView).setVisibility(View.VISIBLE);

                // Load image into ImageView
                InputStream is = getAssets().open("tests/" + assetFilename);
                Bitmap bitmap = BitmapFactory.decodeStream(is);
                // ... (Set bitmap to ImageView)
                myview = findViewById(R.id.MainImageView);
                ((ImageView) myview).setImageBitmap(bitmap);
                //inference
                processImage(bitmap);

            } else if (assetFilename.endsWith(".mp4")) {
                // Load video into VideoView
                String videoPath = getUriFromAsset(assetFilename);
                Log.d("Path: ", videoPath);
                // ... (Set videoPath to VideoView)
                findViewById(R.id.MainVideoView).setVisibility(View.VISIBLE);
                findViewById(R.id.MainImageView).setVisibility(View.GONE);
                //inference
                myview = findViewById(R.id.MainVideoView);
                Uri videoUri = Uri.fromFile(new File(videoPath));
                Log.d("Video uri: ", videoUri.toString());
                playVideo(videoUri);

            }
        } catch (IOException e) {
            Log.e(TAG, "Error loading asset", e);
        }
    }

    private String getUriFromAsset(String assetFilename) throws FileNotFoundException {
        try {
            File outputFile = new File(getFilesDir(), assetFilename);
            AssetFileDescriptor afd = getAssets().openFd("tests/" + assetFilename);
            long totalSize = afd.getLength();
            afd.close();
            try {
                InputStream inputStream = getAssets().open("tests/" + assetFilename);
                OutputStream outputStream = new FileOutputStream(outputFile);
                // Buffer size for copying
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    // Use the API level 33 specific code here
                    inputStream.transferTo(outputStream);
                }
                else {
                    byte[] buffer = new byte[(int) totalSize];  // 1 KB buffer size
                    int bytesRead;

                    // Read from input stream and write to output stream
                    while ((bytesRead = inputStream.read(buffer)) != -1) {
                        outputStream.write(buffer, 0, bytesRead);
                    }
                }
            } catch (IOException e) {
                Log.e(TAG, "Error loading asset", e);
                return null;
            }
            return outputFile.getAbsolutePath();
        } catch (IOException e) {
            Log.e(TAG, "Error loading asset", e);
            return null;
        }
    }

    private void processImage(Bitmap image) {
        checkClearUI();
        new Thread(() -> {
            // ... (Process inference results and update UI)
            if (image != null) {
                // Create a Result object and update the UI
                float percentage =  100;
                Inference(image, percentage);
            }
        }).start();
    }

    private void processVideoFrames(Uri videoUri) {
        checkClearUI();
        //run
        new Thread(() -> {

            MediaMetadataRetriever retriever = new MediaMetadataRetriever();
            retriever.setDataSource(this, videoUri);

            long timestamp = 0L;
            long videoDuration = Long.parseLong(retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION));

            Log.d("Video duration:", String.valueOf(videoDuration));
            int currentTime = 0;

            while (currentTime < videoDuration) {
                currentTime = ((VideoView) myview).getCurrentPosition(); //ms
                ((VideoView) myview).pause();
                Log.d("Current position", String.valueOf(currentTime));
//                timestamp = timestamp + currentTime;
                Log.d("Timestamp", String.valueOf(timestamp));
                Bitmap frame = retriever.getFrameAtTime(currentTime* 1000L);
                // Process frame and calculate probs_spoof

                if (frame != null) {
                    // Create a Result object and update the UI
                    float percentage = (float) currentTime / videoDuration * 100f;
                    Inference(frame, percentage);
                }
                else  {
                    Log.d("Frame", "null");
                }
                ((VideoView) myview).start();
//                ((VideoView) myview).seekTo((int) timestamp);
            }
            try {
                retriever.release();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }).start();


    }

    private void checkClearUI(){
        binding.detectedItemValue3.setVisibility(View.GONE);
        binding.detectedItemValue2.setVisibility(View.GONE);
        binding.detectedItemValue1.setVisibility(View.GONE);
        binding.imageDetection1.setVisibility(View.GONE);
        binding.imageDetection2.setVisibility(View.GONE);
        binding.imageDetection3.setVisibility(View.GONE);
    }


    private void playVideo(Uri videoUri) {
        if (videoUri != null) {
            ((VideoView) myview).setVideoURI(videoUri);
            ((VideoView) myview).setOnPreparedListener(mp -> {
                ((VideoView) myview).start();
                processVideoFrames(videoUri);
            });

            ((VideoView) myview).setOnErrorListener((mp, what, extra) -> {
                Log.e("VideoView", "Error occurred: " + what + ", " + extra);
                return true;
            });
        }
    }


//    private void updateUI(Result result, float percentage) {
//        if (result.detectedScore.isEmpty()) {
//            return ;
//        }
//
//        runOnUiThread(() -> {
////            int probs = (int) (result.detectedScore.get(0) * 100);
//            Log.d("Percentage", String.valueOf((int) percentage));
//            binding.percentMeter.setProgress((int) percentage, true);
//
//            String label = result.detectedIndices.get(0) == 0 ? labelData.get(0) : labelData.get(1);
//            binding.detectedItem1.setText(label);
//            binding.detectedItemValue1.setText(String.format("%.4f%%", result.detectedScore.get(0) * 100));
//
//
//            if (result.detectedIndices.size() > 1) {
//                String label2 = result.detectedIndices.get(1) == 0 ? labelData.get(0) : labelData.get(1);
//                binding.detectedItem2.setText(label2);
//                binding.detectedItemValue2.setText(String.format("%.4f%%", result.detectedScore.get(1) * 100));
//            }
//
//            if (result.detectedIndices.size() > 2) {
//                String label3 = result.detectedIndices.get(2) == 0 ? labelData.get(0) : labelData.get(1);
//                binding.detectedItem3.setText(label3);
//                binding.detectedItemValue3.setText(String.format("%.4f%%", result.detectedScore.get(2) * 100));
//            }
//
//            binding.inferenceTimeValue.setText(result.processTimeMs + "ms");
//        });
//    }
    private void updateUI(DetectionResult[] results, long processTimeMs, float percentage) {
        if (results.length == 0) {
            return ;
        }

        runOnUiThread(() -> {
            Log.d("Percentage", String.valueOf((int) percentage));
            binding.percentMeter.setProgress((int) percentage, true);

            float score = results[0].probs[0];
            String label = score < SCORE_SPOOF ? labelData.get(0) : labelData.get(1);
            binding.detectedItem1.setText(label);
            binding.detectedItemValue1.setText(String.format("%.4f%%", score * 100f));
            binding.detectedItemValue1.setVisibility(View.VISIBLE);
            binding.imageDetection1.setImageBitmap(results[0].faceBitmap);
            binding.imageDetection1.setVisibility(View.VISIBLE);

            if (results.length > 1) {
                float score2 = results[1].probs[0];
                String label2 = score2 < SCORE_SPOOF ? labelData.get(0) : labelData.get(1);
                binding.detectedItem2.setText(label2);
                binding.detectedItemValue2.setText(String.format("%.4f%%", score2 * 100f));
                binding.detectedItemValue2.setVisibility(View.VISIBLE);
                binding.imageDetection2.setImageBitmap(results[1].faceBitmap);
                binding.imageDetection2.setVisibility(View.VISIBLE);
            }

            if (results.length > 2) {
                float score3 = results[2].probs[0];
                String label3 = score3 < SCORE_SPOOF ? labelData.get(0) : labelData.get(1);
                binding.detectedItem3.setText(label3);
                binding.detectedItemValue3.setText(String.format("%.4f%%", score3 * 100f));
                binding.detectedItemValue3.setVisibility(View.VISIBLE);
                binding.imageDetection3.setImageBitmap(results[2].faceBitmap);
                binding.imageDetection3.setVisibility(View.VISIBLE);
            }

            binding.inferenceTimeValue.setText(String.format("%dms", processTimeMs));
        });
    }

    private List<String> readLabels() {
        List<String> labels = new ArrayList<>();
        try (InputStream inputStream = getResources().openRawResource(R.raw.spoofing_label);
             BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream, StandardCharsets.UTF_8))) {

            String line;
            while ((line = reader.readLine()) != null) {
                labels.add(line);
            }
        } catch (IOException e) {
            Log.e(TAG, "Error reading labels", e);
        }
        return labels;
    };

    private String readYolo() throws Exception {
//       // Step 2: Get input stream from resource

        File outputFile = new File(getFilesDir(), "yolov7_hf_v1.onnx");
        // Step 3: Define output file in internal storage
        try {
            InputStream inputStream = getResources().openRawResource(R.raw.yolov7_hf_v1);
            OutputStream outputStream = new FileOutputStream(outputFile);
            // Buffer size for copying
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                // Use the API level 33 specific code here
                inputStream.transferTo(outputStream);
            }
            else {
                AssetFileDescriptor afd = getResources().openRawResourceFd(R.raw.yolov7_hf_v1);
                long totalSize = afd.getLength();
                afd.close();
                Log.d("Total size", String.valueOf(totalSize));
                byte[] buffer = new byte[(int) totalSize];  // 1 KB buffer size
                int bytesRead;

                // Read from input stream and write to output stream
                while ((bytesRead = inputStream.read(buffer)) != -1) {
                    outputStream.write(buffer, 0, bytesRead);
                }
            }
        } catch (IOException e) {
            Log.e(TAG, "Error loading asset", e);
            return null;
        }
        // Step 4: Return the absolute path of the copied file
        return outputFile.getAbsolutePath();
    }

    private String readEmbedder() throws Exception {
//        // Step 2: Get input stream from resource


        File outputFile = new File(getFilesDir(), "embedder.onnx");

        try {
            InputStream inputStream = getResources().openRawResource(R.raw.embedder);
            OutputStream outputStream = new FileOutputStream(outputFile);
            // Buffer size for copying
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                // Use the API level 33 specific code here
                inputStream.transferTo(outputStream);
            }
            else {
                AssetFileDescriptor afd = getResources().openRawResourceFd(R.raw.embedder);
                long totalSize = afd.getLength();
                afd.close();
                byte[] buffer = new byte[(int) totalSize];  // 1 KB buffer size
                int bytesRead;

                // Read from input stream and write to output stream
                while ((bytesRead = inputStream.read(buffer)) != -1) {
                    outputStream.write(buffer, 0, bytesRead);
                }
            }
        } catch (IOException e) {
            Log.e(TAG, "Error loading asset", e);
            return null;
        }
        // Step 4: Return the absolute path of the copied file
        return outputFile.getAbsolutePath();
    }

    private String readExtractor() throws Exception {
        // Step 2: Get input stream from resource


        File outputFile = new File(getFilesDir(), "extractor.onnx");

        // Step 3: Define output file in internal storage
        try {
            InputStream inputStream = getResources().openRawResource(R.raw.extractor);
            OutputStream outputStream = new FileOutputStream(outputFile);
            // Buffer size for copying
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                // Use the API level 33 specific code here
                inputStream.transferTo(outputStream);
            }
            else {
                AssetFileDescriptor afd = getResources().openRawResourceFd(R.raw.extractor);
                long totalSize = afd.getLength();
                afd.close();
                byte[] buffer = new byte[(int) totalSize];  // 1 KB buffer size
                int bytesRead;

                // Read from input stream and write to output stream
                while ((bytesRead = inputStream.read(buffer)) != -1) {
                    outputStream.write(buffer, 0, bytesRead);
                }
            }
        } catch (IOException e) {
            Log.e(TAG, "Error loading asset", e);
            return null;
        }
        // Step 4: Return the absolute path of the copied file
        return outputFile.getAbsolutePath();
    }

    // update UI
    private void Inference(Bitmap frame, float percentage) {

        long time = System.currentTimeMillis();
        // Run inference on the frame
        DetectionResult[] detectionResults = TensorUtils.checkspoof(frame);
        long processTimeMs = System.currentTimeMillis() - time;

        if (detectionResults != null) {
            updateUI(detectionResults, processTimeMs, percentage);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        backgroundExecutor.shutdown();
    }
}