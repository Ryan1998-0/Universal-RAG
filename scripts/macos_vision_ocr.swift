#!/usr/bin/env swift

import AppKit
import Foundation
import Vision

struct OcrPayload: Codable {
    let text: String
    let confidence: Double
    let engine: String
    let languages: [String]
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}

guard CommandLine.arguments.count == 2 else {
    fail("Usage: macos_vision_ocr.swift <image-path>")
}

let imagePath = CommandLine.arguments[1]
guard let image = NSImage(contentsOfFile: imagePath) else {
    fail("Unable to open image: \(imagePath)")
}

var imageRect = NSRect(origin: .zero, size: image.size)
guard let cgImage = image.cgImage(forProposedRect: &imageRect, context: nil, hints: nil) else {
    fail("Unable to create CGImage: \(imagePath)")
}

let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesLanguageCorrection = true
let preferredLanguages = ["zh-Hant", "zh-Hans", "en-US"]
let supportedLanguages = (try? request.supportedRecognitionLanguages()) ?? []
let selectedLanguages = preferredLanguages.filter { supportedLanguages.contains($0) }
if !selectedLanguages.isEmpty {
    request.recognitionLanguages = selectedLanguages
}

do {
    try VNImageRequestHandler(cgImage: cgImage, options: [:]).perform([request])
} catch {
    fail("Vision OCR failed: \(error.localizedDescription)")
}

let observations = (request.results ?? []).sorted { left, right in
    let verticalDifference = left.boundingBox.midY - right.boundingBox.midY
    if abs(verticalDifference) > 0.015 {
        return verticalDifference > 0
    }
    return left.boundingBox.minX < right.boundingBox.minX
}

var lines: [String] = []
var confidences: [Double] = []
for observation in observations {
    guard let candidate = observation.topCandidates(1).first else { continue }
    let text = candidate.string.trimmingCharacters(in: .whitespacesAndNewlines)
    if text.isEmpty { continue }
    lines.append(text)
    confidences.append(Double(candidate.confidence))
}

let averageConfidence = confidences.isEmpty
    ? 0.0
    : confidences.reduce(0.0, +) / Double(confidences.count)
let payload = OcrPayload(
    text: lines.joined(separator: "\n"),
    confidence: averageConfidence,
    engine: "macos-vision",
    languages: selectedLanguages
)

do {
    let output = try JSONEncoder().encode(payload)
    FileHandle.standardOutput.write(output)
    FileHandle.standardOutput.write(Data("\n".utf8))
} catch {
    fail("Unable to encode OCR result: \(error.localizedDescription)")
}
