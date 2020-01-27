package org.example.ahmad.android_test;

import android.content.Intent;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;


/**
 * The Application is developed in partial fulfillment of
 * the course Research Project(MOD3-03) in M.Eng Embedded Systems for Mechatronics.
 *
 * @author ahmad.mirza001@stud.fh-dortmund.de (Ahmad H. Mirza)
 * MatrikelNo : 7104716
 */


/**
 * MainActivity is the first class that is started upon application's launch
 */
public class MainActivity extends AppCompatActivity {

    private static final String TAG = "Main Activity";

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);
    }

    /**
     * This function is Triggered on the button press from the user. Creates an Intent object
     * to start AvatarCameraActivity
     *
     * @param view object - basic building blocks of User Interface(UI) elements in Android. They are
     *             simple rectangle box which responds to the user's actions, e.g. EditText, Button,
     *             CheckBox etc. View refers to the android.view.View class, which is the base
     *             class of all UI classes.
     */
    public void avatarStart(View view) {

        Intent intent = new Intent(this, AvatarCameraActivity.class);
        startActivity(intent);
    }

}
