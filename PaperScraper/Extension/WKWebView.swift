//
//  WKWebView.swift
//  PaperScraper
//
//  Created by Khislatjon Valijonov on 20/07/2025.
//

import WebKit

extension WKWebView {
    func loadRequestAsync(_ request: URLRequest) async throws -> WKNavigation {
        return try await withCheckedThrowingContinuation { continuation in
            let nav = self.load(request)
            DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                // Wait a few seconds to ensure full render
                if let nav = nav {
                    continuation.resume(returning: nav)
                } else {
                    continuation.resume(throwing: URLError(.cannotLoadFromNetwork))
                }
            }
        }
    }

    func createPDF() async throws -> Data {
        return try await withCheckedThrowingContinuation { continuation in
            let config = WKPDFConfiguration()
            self.createPDF(configuration: config) { result in
                switch result {
                case .success(let data):
                    continuation.resume(returning: data)
                case .failure(let error):
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}
