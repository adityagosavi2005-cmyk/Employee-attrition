import 'dart:convert';
import 'package:http/http.dart' as http;

Future<double> predictRisk(Map<String, dynamic> data) async {

  final response = await http.post(
    Uri.parse("http://localhost:8000/predict"),
    headers: {"Content-Type": "application/json"},
    body: jsonEncode(data),
  );

  final result = jsonDecode(response.body);

  return result["risk_score"];
}