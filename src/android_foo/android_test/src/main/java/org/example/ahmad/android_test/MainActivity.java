package org.example.ahmad.android_test;

import android.content.Intent;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;


/**
 * @author ahmad.mirza001@stud.fh-dortmund.de (Ahmad H. Mirza)
 */
public class MainActivity extends AppCompatActivity {

    private static final String TAG = "Main Activity";

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

    }

    public void avatarStart(View view) {

        Intent intent = new Intent(this, AvatarCameraActivity.class);
        startActivity(intent);
    }

}
