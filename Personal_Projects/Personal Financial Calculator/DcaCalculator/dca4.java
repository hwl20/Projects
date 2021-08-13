package DcaCalculator;

import java.util.Scanner;

public class dca4 {
    public static void tryout4(int changes) {
        Scanner scanner = new Scanner(System.in);
        double[] storage = new double[changes];
        double[] yearList = new double[changes];

        System.out.print("Type in amount put in monthly initially: ");
        int monthlyAmount = scanner.nextInt();
        double yearlyAmount = monthlyAmount * 12;
        storage[0] = yearlyAmount;

        System.out.format("Time horizon for $%d: ", monthlyAmount);
        double year = scanner.nextInt();
        yearList[0] = year;

        for (int i=1; i<changes; i++){
            System.out.print("\nType in amount put in monthly next: ");
            int monthHolder = scanner.nextInt();
            double yearHolder = monthHolder * 12;
            storage[i] = yearHolder;

            System.out.format("Time horizon for $%d: ", monthHolder);
            double timeHolder = scanner.nextInt();
            yearList[i] = timeHolder;
        }

        System.out.print("Interest per annum (%): ");
        double interest = scanner.nextDouble()/100;

        double amount = 20000;
        for (int counter=0; counter<storage.length; counter++) {
            for (int i = 1; i <= yearList[counter]; i++) {
                amount = amount * (1 + interest) + storage[counter];
            }
        }
        double totalNumOfYears = 0;
        for (int counter=0; counter<storage.length; counter++){
            totalNumOfYears += yearList[counter];
        }
        System.out.format("\nAmount at the end of %.0f years is: $%.2f", totalNumOfYears, amount);
    }
}
