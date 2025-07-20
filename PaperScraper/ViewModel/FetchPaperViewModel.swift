//
//  FetchPaperViewModel.swift
//  PaperScraper
//
//  Created by Khislatjon Valijonov on 20/07/2025.
//

import SwiftUI
import Foundation
import WebKit

@Observable class FetchPaperViewModel {
    var papers: [ArxivPaper] = []
    
    func downloadArxivPDFs(maxResults: Int = 5) async {
        let topics = ["NLP", "graph neural networks"]
        let query = buildArxivQuery(from: topics)
        let encodedQuery = query.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed)!
        
        do {
            self.papers = try await fetchArxivPapers(encodedQuery: encodedQuery, maxResults: maxResults)

            for (_, paper) in papers.enumerated() {
                try await Task.sleep(nanoseconds: 5_000_000_000) // 5 seconds delay

                guard let pdfURL = getPDFURL(from: paper.link) else {
                    print("❌ Could not get PDF URL from: \(paper.link)")
                    continue
                }

                let sanitizedTitle = paper.title
                    .replacingOccurrences(of: "[^a-zA-Z0-9_\\-]", with: "_", options: .regularExpression)
                    .prefix(50) // Truncate to prevent filename issues

                let filePath = FileManager.default.temporaryDirectory
                    .appendingPathComponent("\(sanitizedTitle).pdf")

                do {
                    let data = try await downloadPDF(from: pdfURL)
                    try data.write(to: filePath)
                    print("✅ Saved: \(filePath.path)")
                } catch {
                    print("❌ Failed to download \(pdfURL): \(error)")
                }
            }

        } catch {
            print("❌ Error fetching papers: \(error)")
        }
    }
    
    private func buildArxivQuery(from topics: [String]) -> String {
        guard !topics.isEmpty else { return "all:" }

        let escapedTerms = topics.map { "\"\($0)\"" }
        let orQuery = escapedTerms.joined(separator: " OR ")
        return "all:(\(orQuery))"
    }
    
    private func fetchArxivPapers(encodedQuery: String, maxResults: Int = 5) async throws -> [ArxivPaper] {
        let urlString = "https://export.arxiv.org/api/query?search_query=\(encodedQuery)&start=0&max_results=\(maxResults)"
        
        guard let url = URL(string: urlString) else {
            throw URLError(.badURL)
        }

        let (data, _) = try await URLSession.shared.data(from: url)
        let parser = ArxivXMLParser()
        return parser.parse(data: data)
    }
    
    private func getPDFURL(from absLink: String) -> URL? {
        guard absLink.contains("/abs/") else { return nil }
        let pdfLink = absLink
            .replacingOccurrences(of: "http", with: "https")
            .replacingOccurrences(of: "/abs/", with: "/pdf/") + ".pdf"
        return URL(string: pdfLink)
    }

    private func downloadPDF(from url: URL) async throws -> Data {
        let (data, response) = try await URLSession.shared.data(from: url)

        guard let httpResp = response as? HTTPURLResponse, httpResp.statusCode == 200 else {
            throw URLError(.badServerResponse)
        }
        
        return data
    }
}
