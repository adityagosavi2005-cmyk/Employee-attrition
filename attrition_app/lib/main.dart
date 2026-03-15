import 'package:flutter/material.dart';
import 'api_service.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: "Attrition Predictor",
      theme: ThemeData.dark(),
      home: PredictionScreen(),
    );
  }
}

class PredictionScreen extends StatefulWidget {
  @override
  _PredictionScreenState createState() => _PredictionScreenState();
}

class _PredictionScreenState extends State<PredictionScreen> {

  final ageController = TextEditingController();
  final serviceController = TextEditingController();

  double? riskScore;

  void predict() async {

    Map<String, dynamic> data = {
      "age": int.parse(ageController.text),
      "length_of_service": int.parse(serviceController.text),
      "city_name": "Vancouver",
      "department_name": "Meats",
      "job_title": "Cashier",
      "store_name": "Store 1",
      "gender_short": "M",
      "BUSINESS_UNIT": "Retail",
      "STATUS_YEAR": 2015
    };

    double result = await predictRisk(data);

    setState(() {
      riskScore = result;
    });
  }

  @override
  Widget build(BuildContext context) {

    return Scaffold(
      appBar: AppBar(
        title: Text("Employee Attrition Predictor"),
      ),

      body: Padding(
        padding: EdgeInsets.all(20),

        child: Column(
          children: [

            TextField(
              controller: ageController,
              keyboardType: TextInputType.number,
              decoration: InputDecoration(
                labelText: "Age",
                border: OutlineInputBorder(),
              ),
            ),

            SizedBox(height: 20),

            TextField(
              controller: serviceController,
              keyboardType: TextInputType.number,
              decoration: InputDecoration(
                labelText: "Years of Service",
                border: OutlineInputBorder(),
              ),
            ),

            SizedBox(height: 20),

            ElevatedButton(
              onPressed: predict,
              child: Text("Predict Attrition Risk"),
            ),

            SizedBox(height: 30),

            if (riskScore != null)
              Text(
                "Risk Score: ${(riskScore! * 100).toStringAsFixed(1)}%",
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                  color: riskScore! > 0.7 ? Colors.red : Colors.green,
                ),
              ),
          ],
        ),
      ),
    );
  }
}